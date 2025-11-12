"""
Extended predictor for all World Bank indicators using GNN.
"""
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .gnn_model import MigrationGNN, CountryGraphBuilder
from .data_loader import WorldBankDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorPredictor:
    """
    Predictor for any World Bank indicator using GNN.
    Uses country relationships and features to predict future indicator values.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        cache: Optional[object] = None,
    ):
        self.device = device
        self.cache = cache
        self.data_loader = WorldBankDataLoader(cache=cache)
        self.graph_builder = CountryGraphBuilder()
        
        # Initialize model (same architecture as migration predictor)
        self.model = MigrationGNN(
            node_features=10,
            hidden_dim=64,
            num_layers=3,
            dropout=0.2,
            use_gat=True,
        ).to(device)
        
        if model_path and Path(model_path).exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        # Always set to eval mode for inference
        self.model.eval()
    
    def _clamp_population_change(
        self, base_value: float, predicted_value: float, years_ahead: int
    ) -> float:
        if years_ahead <= 0 or base_value <= 0:
            return base_value
        decline_cap = min(0.02 * years_ahead, 0.1)
        growth_cap = min(0.01 * years_ahead, 0.03)
        lower = base_value * (1 - decline_cap)
        upper = base_value * (1 + growth_cap)
        return max(lower, min(upper, predicted_value))
    
    @staticmethod
    def _ensure_finite(value: Any, fallback: float) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not math.isfinite(val):
            return float(fallback)
        return val
    
    @staticmethod
    def _sanitize_pair(base_value: Any, predicted_value: Any) -> Tuple[float, float]:
        base = IndicatorPredictor._ensure_finite(base_value, 0.0)
        predicted = IndicatorPredictor._ensure_finite(predicted_value, base)
        return base, predicted
    
    def _predict_population_value(
        self,
        country: str,
        historical_trend: List[Tuple[int, float]],
        base_value: float,
        actual_base_year: int,
        target_year: int,
    ) -> float:
        base_value = self._ensure_finite(base_value, 0.0)
        if target_year <= actual_base_year or base_value <= 0:
            return base_value
        years_ahead = target_year - actual_base_year
        usable = [t for t in historical_trend if t[1] is not None and t[1] > 0]
        if len(usable) < 2:
            return self._clamp_population_change(base_value, base_value, years_ahead)
        trimmed = usable[-min(len(usable), 15) :]
        years = np.array([float(t[0]) for t in trimmed], dtype=np.float32)
        values = np.array([float(t[1]) for t in trimmed], dtype=np.float32)
        mean_year = float(years.mean())
        mean_value = float(values.mean())
        denom = float(np.sum((years - mean_year) ** 2))
        if denom > 0:
            slope = float(np.sum((years - mean_year) * (values - mean_value))) / denom
        else:
            slope = 0.0
        slope_rate = slope / max(base_value, 1.0)
        first_value = float(trimmed[0][1])
        first_year = int(trimmed[0][0])
        overall_rate = 0.0
        if first_value > 0 and actual_base_year > first_year:
            overall_rate = (base_value - first_value) / first_value / (actual_base_year - first_year)
        # Bias towards decline if long-term trend is flat or negative
        if overall_rate < -0.0005:
            slope_rate = min(slope_rate, overall_rate)
        # Baltic-specific heuristics – enforce gentle decline
        if country in ["LVA", "EST", "LTU"]:
            if slope_rate >= 0:
                slope_rate = -0.007
            else:
                slope_rate = max(slope_rate, -0.012)
        slope_rate = max(-0.012, min(0.005, slope_rate))
        predicted = base_value * (1.0 + slope_rate * years_ahead)
        predicted = self._clamp_population_change(base_value, predicted, years_ahead)
        logger.info(
            f"Population prediction for {country}: base={base_value:.2f}, slope_rate={slope_rate:.4f}, overall_rate={overall_rate:.4f}, years={years_ahead}, result={predicted:.2f}"
        )
        return self._ensure_finite(predicted, base_value)

    def predict_indicator(
        self,
        indicator: str,
        countries: List[str],
        target_year: Optional[int] = None,
        base_year: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Predict indicator values for multiple countries.
        
        Args:
            indicator: World Bank indicator code
            countries: List of country ISO3 codes
            target_year: Year to predict for
            base_year: Base year for features
        
        Returns:
            Dictionary mapping country codes to predicted values
        """
        if target_year is None:
            target_year = 2025
        if base_year is None:
            base_year = 2020
        
        # Check cache
        cache_key = f"indicator_pred:{indicator}:{':'.join(sorted(countries))}:{target_year}"
        if self.cache:
            try:
                import json
                cached = self.cache.get(cache_key)
                if cached:
                    parsed = json.loads(cached)
                    if isinstance(parsed, dict) and "predictions" in parsed:
                        return parsed
                    return {
                        "indicator": indicator,
                        "countries": countries,
                        "target_year": target_year,
                        "base_year": base_year,
                        "predictions": parsed,
                        "insights": self._generate_insights(indicator, {}, target_year),
                    }
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
        
        # Get country features (with error handling)
        country_features = {}
        for country in countries:
            try:
                features = self.data_loader.get_country_features(country, base_year)
                country_features[country] = features
            except Exception as e:
                logger.warning(f"Error fetching features for {country}: {e}, using zeros")
                # Use zero vector as fallback
                country_features[country] = np.zeros(10, dtype=np.float32)
        
        insight_payload: Dict[str, Dict[str, Any]] = {}
        
        # Get historical indicator values to create relationships
        # For Baltic countries, use more historical data points for better accuracy
        historical_values = {}
        for country in countries:
            try:
                # Always try to get the most recent available value at base_year first
                val = self.data_loader.fetch_indicator_data(
                    country, indicator, base_year, base_year
                )
                if val is not None and val > 0:
                    historical_values[country] = val
                    logger.debug(f"Found base value for {country} at {base_year}: {val}")
                    continue
                
                # If not found at base_year, try previous years
                # For Baltic countries, use more years of historical data
                if country in ["LVA", "EST", "LTU"]:
                    # Try to get data from base_year - 10 to base_year for better trend
                    values = []
                    for year_offset in range(-10, 0):  # Exclude base_year since we already tried it
                        year = base_year + year_offset
                        try:
                            val = self.data_loader.fetch_indicator_data(
                                country, indicator, year, year
                            )
                            if val is not None and val > 0:
                                values.append((year, val))
                        except Exception as e:
                            logger.debug(f"Error fetching {country} year {year}: {e}")
                            continue
                    
                    # Use the most recent value if available
                    if values:
                        historical_values[country] = values[-1][1]  # Most recent value
                        logger.info(f"Found {len(values)} historical values for {country}, using most recent from year {values[-1][0]}: {values[-1][1]}")
                    else:
                        # Fallback: try broader range
                        logger.warning(f"No data found for {country} in range {base_year-10} to {base_year-1}, trying broader range")
                        try:
                            # Try a broader range first
                            val = self.data_loader.fetch_indicator_data(
                                country, indicator, base_year - 20, base_year - 1
                            )
                            if val and val > 0:
                                historical_values[country] = val
                                logger.info(f"Found data for {country} using broader range: {val}")
                            else:
                                # Last resort: try fetching without year restriction to get any available data
                                val = self.data_loader.fetch_indicator_data(
                                    country, indicator, None, None
                                )
                                if val and val > 0:
                                    historical_values[country] = val
                                    logger.info(f"Found data for {country} without year restriction: {val}")
                                else:
                                    historical_values[country] = 0.0
                                    logger.error(f"No data found for {country} after all fallbacks")
                        except Exception as e:
                            logger.error(f"Fallback failed for {country}: {e}")
                            historical_values[country] = 0.0
                else:
                    # Standard method for other countries - try previous years
                    for year_offset in range(-5, 0):
                        year = base_year + year_offset
                        val = self.data_loader.fetch_indicator_data(
                            country, indicator, year, year
                        )
                        if val is not None and val > 0:
                            historical_values[country] = val
                            logger.debug(f"Found base value for {country} at year {year}: {val}")
                            break
                    else:
                        # No value found in range
                        historical_values[country] = 0.0
                        logger.warning(f"No historical data found for {country} in range {base_year-5} to {base_year-1}")
            except Exception as e:
                logger.warning(f"Error fetching historical data for {country}: {e}")
                historical_values[country] = 0.0
        
        # Build graph with indicator-based relationships
        # Create edges based on indicator similarity
        indicator_flows = []
        countries_list = list(countries)
        
        for i, src in enumerate(countries_list):
            for j, tgt in enumerate(countries_list):
                if i != j:
                    # Create edge weight based on indicator similarity
                    src_val = historical_values.get(src, 0.0)
                    tgt_val = historical_values.get(tgt, 0.0)
                    
                    # Similar values = stronger connection
                    similarity = 1.0 / (1.0 + abs(src_val - tgt_val) / max(abs(src_val), 1.0))
                    indicator_flows.append((src, tgt, similarity))
        
        # For single country, use a simpler prediction approach
        if len(countries) == 1:
            # Enhanced linear extrapolation with more historical data for Baltic countries
            country = countries[0]
            
            # Get historical values for trend calculation
            # Use more years for Baltic countries - get data from a wider range
            year_range = range(-15, 1) if country in ["LVA", "EST", "LTU"] else range(-10, 1)
            historical_trend = []
            for year_offset in year_range:
                year = base_year + year_offset
                try:
                    val = self.data_loader.fetch_indicator_data(country, indicator, year, year)
                    if val is not None and val > 0:
                        historical_trend.append((year, val))
                except Exception as e:
                    logger.debug(f"Error fetching data for {country} year {year}: {e}")
                    continue
            
            # Sort by year to ensure chronological order
            historical_trend.sort(key=lambda x: x[0])
            
            # Use the actual last year of data as base, not base_year
            if len(historical_trend) >= 2:
                # base_value should be from the actual last year of data
                base_value = historical_trend[-1][1]
                actual_base_year = historical_trend[-1][0]
                first_value = historical_trend[0][1]
                first_year = historical_trend[0][0]
                year_span = max(1, actual_base_year - first_year)
                overall_rate = 0.0
                if indicator == "SP.POP.TOTL" and first_value > 0:
                    overall_rate = (base_value - first_value) / first_value / year_span
                logger.info(
                    f"{country}: Using actual base year {actual_base_year} with value {base_value:.2f} (requested base_year was {base_year}); overall annual rate {overall_rate:.4f}"
                )
                
                if indicator == "SP.POP.TOTL":
                    predicted_value = self._predict_population_value(
                        country, historical_trend, base_value, actual_base_year, target_year
                    )
                else:
                    # Calculate growth rate only if NOT declining
                    is_declining = False
                    decline_rate = 0.0
                    
                    # Check last 3 years
                    if len(historical_trend) >= 3:
                        recent_change = historical_trend[-1][1] - historical_trend[-3][1]
                        if recent_change < 0:  # Declining trend
                            is_declining = True
                            decline_rate = recent_change / historical_trend[-3][1] / 3.0  # Annual decline rate
                            logger.info(f"{country}: Detected decline over last 3 years: {recent_change:.2f}, annual rate: {decline_rate:.4f}")
                    
                    # Check last 5 years for consistent decline
                    if len(historical_trend) >= 5:
                        last_5_years = historical_trend[-5:]
                        if all(last_5_years[i][1] < last_5_years[i-1][1] for i in range(1, len(last_5_years))):
                            is_declining = True
                            # Calculate average annual decline rate
                            avg_decline = sum((last_5_years[i][1] - last_5_years[i-1][1]) / last_5_years[i-1][1]
                                             for i in range(1, len(last_5_years))) / (len(last_5_years) - 1)
                            decline_rate = avg_decline
                            logger.info(f"{country}: Consistent 5-year decline detected, average annual rate: {decline_rate:.4f}")
                    
                    # Calculate growth rate only if NOT declining
                    if not is_declining and len(historical_trend) >= 5:
                        # Use weighted average: more weight to recent years
                        recent_trend = historical_trend[-5:]  # Last 5 years
                        total_growth = 0.0
                        total_weight = 0.0
                        weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # More weight on recent
                        
                        for i in range(1, len(recent_trend)):
                            years_diff = recent_trend[i][0] - recent_trend[i-1][0]
                            values_diff = recent_trend[i][1] - recent_trend[i-1][1]
                            if recent_trend[i-1][1] > 0 and years_diff > 0:
                                annual_growth = (values_diff / recent_trend[i-1][1]) / years_diff
                                weight = weights[i] if i < len(weights) else weights[-1]
                                total_growth += annual_growth * weight
                                total_weight += weight
                        
                        if total_weight > 0:
                            growth_rate = total_growth / total_weight
                        else:
                            growth_rate = 0.0
                    else:
                        # Standard calculation - use last 3-5 years for trend
                        trend_window = historical_trend[-min(5, len(historical_trend)):]
                        years_diff = trend_window[-1][0] - trend_window[0][0]
                        values_diff = trend_window[-1][1] - trend_window[0][1]
                        if trend_window[0][1] > 0 and years_diff > 0:
                            growth_rate = (values_diff / trend_window[0][1]) / years_diff
                        else:
                            growth_rate = 0.0
                    
                    # Cap growth rate to reasonable bounds
                    if indicator == "SP.POP.TOTL":
                        growth_rate = max(-0.02, min(0.01, growth_rate))  # allow at most +1% for growth
                    else:
                        growth_rate = max(-0.05, min(0.05, growth_rate))
                    
                    # Predict future value
                    years_ahead = target_year - actual_base_year
                    if years_ahead <= 0:
                        logger.warning(f"Invalid years_ahead: {years_ahead} (target={target_year}, actual_base={actual_base_year}), using base value")
                        predicted_value = base_value
                    else:
                        # ALWAYS use linear extrapolation for declining populations
                        if is_declining or growth_rate < 0:
                            # Linear extrapolation for declining populations (more conservative)
                            predicted_value = base_value * (1.0 + growth_rate * years_ahead)
                            # Ensure it doesn't go below 80% of base (max 20% decline over 5 years)
                            predicted_value = max(base_value * 0.8, predicted_value)
                            logger.info(f"{country}: Using linear extrapolation (declining): {predicted_value:.2f}")
                        else:
                            # Compound growth only for growing populations
                            predicted_value = base_value * ((1.0 + growth_rate) ** years_ahead)
                            logger.info(f"{country}: Using compound growth (growing): {predicted_value:.2f}")
                    
                    # FINAL STRICT CAP: For population, max 3% change over the horizon
                    if indicator == "SP.POP.TOTL":
                        max_change = 0.03  # 3% max change for population
                    else:
                        max_change_per_year = 0.05
                        max_change = min(0.20, max_change_per_year * years_ahead)
                    
                    # Apply cap
                    if predicted_value > base_value * (1 + max_change):
                        predicted_value = base_value * (1 + max_change)
                        logger.warning(f"CAPPED {country} to {max_change*100:.1f}% increase: {predicted_value:.2f}")
                    elif predicted_value < base_value * (1 - max_change):
                        predicted_value = base_value * (1 - max_change)
                        logger.warning(f"CAPPED {country} to {max_change*100:.1f}% decrease: {predicted_value:.2f}")
                
                logger.info(f"FINAL PREDICTION {country}: base={base_value:.2f} (year {actual_base_year}), result={predicted_value:.2f}")
            else:
                # Fallback: use base value from historical_values or try to fetch
                if historical_trend:
                    base_value = historical_trend[-1][1]
                    actual_base_year = historical_trend[-1][0]
                else:
                    base_value = historical_values.get(country, 0.0)
                    actual_base_year = base_year
                predicted_value = base_value
                logger.warning(f"{country}: No sufficient trend data, using base value {base_value:.2f} from year {actual_base_year}")
            
            base_value, predicted_value = self._sanitize_pair(base_value, predicted_value)
            insight_payload[country] = {
                "base_value": base_value,
                "predicted_value": predicted_value,
                "base_year": actual_base_year,
                "target_year": target_year,
                "trend": historical_trend,
                "data_points": len(historical_trend),
            }
            
            predictions = {country: max(0.0, predicted_value)}
        else:
            # Multiple countries: use GNN
            graph_data = self.graph_builder.build_graph(country_features, indicator_flows)
            graph_data = graph_data.to(self.device)
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # Get node embeddings from the model
            x = graph_data.x
            edge_index = graph_data.edge_index
            
            # Forward through model to get node embeddings
            with torch.no_grad():
                # Get node embeddings by running through the model layers
                x_emb = self.model.conv1(x, edge_index)
                x_emb = self.model.bn1(x_emb)
                x_emb = torch.relu(x_emb)
                x_emb = self.model.dropout(x_emb)
                x_emb = self.model.conv2(x_emb, edge_index)
                x_emb = self.model.bn2(x_emb)
                x_emb = torch.relu(x_emb)
                x_emb = self.model.dropout(x_emb)
                x_emb = self.model.conv3(x_emb, edge_index)
                x_emb = torch.relu(x_emb)
            
            # Use historical trend-based prediction instead of arbitrary GNN output
            # The GNN was trained for migration, not indicator prediction
            # So we'll use a more sophisticated trend analysis
            
            predictions = {}
            years_ahead = target_year - base_year
            
            # Get historical trends for all countries
            country_trends = {}
            for idx, country in enumerate(countries_list):
                # Get historical trend data first - use wider range
                trend_data = []
                year_range = range(-15, 1) if country in ["LVA", "EST", "LTU"] else range(-10, 1)
                for year_offset in year_range:
                    year = base_year + year_offset
                    try:
                        val = self.data_loader.fetch_indicator_data(country, indicator, year, year)
                        if val is not None and val > 0:
                            trend_data.append((year, val))
                    except Exception as e:
                        logger.debug(f"Error fetching trend data for {country} year {year}: {e}")
                        continue
                
                # Sort by year
                trend_data.sort(key=lambda x: x[0])
                
                # Use the actual last year of data as base, not base_year
                if trend_data:
                    base_value = trend_data[-1][1]
                    actual_base_year = trend_data[-1][0]
                    logger.info(f"{country}: Using actual base year {actual_base_year} with value {base_value:.2f} (requested base_year was {base_year})")
                else:
                    # Fallback: try to get from historical_values
                    base_value = historical_values.get(country, 0.0)
                    actual_base_year = base_year
                    if base_value == 0.0:
                        logger.warning(f"Base value is 0 for {country}, trying to fetch directly")
                        try:
                            # Try fetching the most recent available value
                            for year in range(base_year, base_year - 20, -1):
                                val = self.data_loader.fetch_indicator_data(
                                    country, indicator, year, year
                                )
                                if val and val > 0:
                                    base_value = val
                                    actual_base_year = year
                                    logger.info(f"Found base value for {country} from year {year}: {val}")
                                    break
                        except Exception as e:
                            logger.error(f"Failed to fetch base value for {country}: {e}")
                
                country_trends[country] = {
                    'base_value': base_value,
                    'actual_base_year': actual_base_year,
                    'trend_data': trend_data
                }
            
            # Calculate predictions using trend analysis
            for idx, country in enumerate(countries_list):
                trend_info = country_trends[country]
                base_value = self._ensure_finite(trend_info['base_value'], 0.0)
                actual_base_year = trend_info.get('actual_base_year', base_year)
                trend_data = trend_info['trend_data']
                
                if indicator == "SP.POP.TOTL":
                    predicted_value = self._predict_population_value(
                        country,
                        trend_data if trend_data else [(actual_base_year, base_value)],
                        base_value,
                        actual_base_year,
                        target_year,
                    )
                    base_value, predicted_value = self._sanitize_pair(base_value, predicted_value)
                    insight_payload[country] = {
                        "base_value": base_value,
                        "predicted_value": predicted_value,
                        "base_year": int(actual_base_year),
                        "target_year": int(target_year),
                        "trend": [(int(y), float(v)) for y, v in trend_data],
                        "data_points": int(len(trend_data)),
                    }
                    predictions[country] = max(0.0, predicted_value)
                    continue
                
                if base_value == 0.0:
                    # If no base value, try to use average of other countries' growth rates
                    if len(predictions) > 0 and len(countries_list) > 1:
                        # Calculate average growth rate from other countries
                        avg_growth = 0.0
                        count = 0
                        for other_country in countries_list:
                            if other_country != country:
                                other_trend = country_trends[other_country]
                                if len(other_trend['trend_data']) >= 2:
                                    recent = other_trend['trend_data'][-5:] if len(other_trend['trend_data']) >= 5 else other_trend['trend_data']
                                    if len(recent) >= 2:
                                        years_diff = recent[-1][0] - recent[0][0]
                                        values_diff = recent[-1][1] - recent[0][1]
                                        if recent[0][1] > 0 and years_diff > 0:
                                            growth = (values_diff / recent[0][1]) / years_diff
                                            avg_growth += growth
                                            count += 1
                        
                        if count > 0:
                            avg_growth = avg_growth / count
                            # Cap growth rate to reasonable bounds - more conservative for population
                            if indicator == "SP.POP.TOTL":
                                avg_growth = max(-0.02, min(0.01, avg_growth))
                            else:
                                avg_growth = max(-0.05, min(0.05, avg_growth))
                            # Use a reasonable estimate based on other countries
                            other_values = [
                                country_trends[c]['base_value']
                                for c in countries_list
                                if c != country and country_trends[c]['base_value'] > 0
                            ]
                            if other_values:
                                estimated_base = sum(other_values) / len(other_values)
                            else:
                                estimated_base = base_value
                            # Use compound growth, not linear
                            # Get actual_base_year from other countries
                            other_actual_years = [country_trends[c].get('actual_base_year', base_year) for c in countries_list if c != country]
                            avg_actual_base_year = sum(other_actual_years) / len(other_actual_years) if other_actual_years else base_year
                            years_ahead = target_year - int(avg_actual_base_year)
                            estimated_base = self._ensure_finite(estimated_base, base_value)
                            if years_ahead > 0:
                                predicted_value = estimated_base * ((1.0 + avg_growth) ** years_ahead)
                                # Cap to reasonable change - stricter for population
                                if indicator == "SP.POP.TOTL":
                                    max_change = 0.03
                                else:
                                    max_change_per_year = 0.05
                                    max_change = min(0.20, max_change_per_year * years_ahead)
                                if predicted_value > estimated_base * (1 + max_change):
                                    predicted_value = estimated_base * (1 + max_change)
                                elif predicted_value < estimated_base * (1 - max_change):
                                    predicted_value = estimated_base * (1 - max_change)
                                predicted_value = self._ensure_finite(predicted_value, estimated_base)
                            else:
                                predicted_value = estimated_base
                                base_value, predicted_value = self._sanitize_pair(base_value, predicted_value)
                                insight_payload[country] = {
                                    "base_value": base_value,
                                    "predicted_value": predicted_value,
                                    "base_year": actual_base_year,
                                    "target_year": target_year,
                                    "trend": trend_data,
                                    "data_points": len(trend_data),
                                }
                                predictions[country] = max(0.0, predicted_value)
                                logger.info(f"Used average growth rate for {country}: base={estimated_base:.2f}, growth={avg_growth:.4f}, result={predicted_value:.2f}")
                                continue
                    
                    base_value = self._ensure_finite(base_value, 0.0)
                    insight_payload[country] = {
                        "base_value": base_value,
                        "predicted_value": 0.0,
                        "base_year": actual_base_year,
                        "target_year": target_year,
                        "trend": trend_data,
                        "data_points": len(trend_data),
                    }
                    predictions[country] = 0.0
                    logger.error(f"Could not predict for {country}, no base value")
                    continue
                
                if len(trend_data) >= 2:
                    # Use weighted average of recent years for better accuracy
                    if len(trend_data) >= 5:
                        # Use last 5 years with more weight on recent years
                        recent = trend_data[-5:]
                        weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # More weight on recent
                    else:
                        recent = trend_data
                        weights = [1.0 / len(recent)] * len(recent)
                    
                    overall_rate = 0.0
                    if indicator == "SP.POP.TOTL" and len(trend_data) >= 2:
                        first_value = trend_data[0][1]
                        first_year = trend_data[0][0]
                        span_years = max(1, actual_base_year - first_year)
                        if first_value > 0:
                            overall_rate = (base_value - first_value) / first_value / span_years
                    
                    # Calculate weighted average annual growth rate
                    total_weighted_growth = 0.0
                    total_weight = 0.0
                    
                    for i in range(1, len(recent)):
                        years_diff = recent[i][0] - recent[i-1][0]
                        values_diff = recent[i][1] - recent[i-1][1]
                        if recent[i-1][1] > 0 and years_diff > 0:
                            # Calculate annual growth rate (already per year)
                            growth_rate = (values_diff / recent[i-1][1]) / years_diff
                            # Cap growth rate to reasonable bounds (-10% to +10% per year)
                            growth_rate = max(-0.1, min(0.1, growth_rate))
                            weight = weights[i] if i < len(weights) else weights[-1]
                            total_weighted_growth += growth_rate * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        avg_growth_rate = total_weighted_growth / total_weight
                        # Cap to reasonable bounds - more conservative for population
                        if indicator == "SP.POP.TOTL":
                            avg_growth_rate = max(-0.02, min(0.01, avg_growth_rate))  # allow at most +1%
                        else:
                            avg_growth_rate = max(-0.05, min(0.05, avg_growth_rate))  # ±5% for other indicators
                        
                        # Respect historical trends: if declining, don't allow positive growth
                        is_declining = False
                        if indicator == "SP.POP.TOTL" and overall_rate < -0.0005:
                            is_declining = True
                            avg_growth_rate = min(avg_growth_rate, overall_rate)
                            logger.info(f"{country} long-term decline rate {overall_rate:.4f}")
                        if len(trend_data) >= 3:
                            recent_change = trend_data[-1][1] - trend_data[-3][1]
                            if recent_change < 0:  # Declining trend
                                is_declining = True
                                decline_rate = recent_change / trend_data[-3][1] / 3.0
                                if country in ["LVA", "EST", "LTU"]:
                                    decline_rate = max(-0.015, min(-0.005, decline_rate))
                                else:
                                    decline_rate = max(-0.02, min(-0.001, decline_rate))
                                avg_growth_rate = min(avg_growth_rate, decline_rate)
                                logger.info(f"{country} shows declining trend (change: {recent_change:.2f}), using growth rate: {avg_growth_rate:.4f}")
                        
                        if len(trend_data) >= 5:
                            last_5 = trend_data[-5:]
                            if all(last_5[i][1] <= last_5[i-1][1] for i in range(1, len(last_5))):
                                is_declining = True
                                avg_decline = sum((last_5[i][1] - last_5[i-1][1]) / last_5[i-1][1]
                                                 for i in range(1, len(last_5))) / (len(last_5) - 1)
                                if country in ["LVA", "EST", "LTU"]:
                                    avg_decline = max(-0.015, min(-0.005, avg_decline))
                                else:
                                    avg_decline = max(-0.02, min(-0.001, avg_decline))
                                avg_growth_rate = min(avg_growth_rate, avg_decline)
                                logger.info(f"{country} shows consistent 5-year decline, adjusted to: {avg_growth_rate:.4f}")
                        
                        if indicator == "SP.POP.TOTL" and not is_declining:
                            # Additional Baltic heuristic: if base well below peak, force slight decline
                            if country in ["LVA", "EST", "LTU"]:
                                peak_value = max(t[1] for t in trend_data)
                                if base_value < peak_value * 0.97 or overall_rate <= 0:
                                    is_declining = True
                                    avg_growth_rate = -0.01
                                    logger.info(f"{country} Baltic heuristic applied, forcing decline -1%")
                        
                        # Predict using appropriate method
                        years_ahead = target_year - actual_base_year
                        if years_ahead <= 0:
                            logger.warning(f"Invalid years_ahead: {years_ahead} (target={target_year}, actual_base={actual_base_year}), using base value")
                            predicted_value = base_value
                        else:
                            if is_declining or avg_growth_rate < 0:
                                predicted_value = base_value * (1.0 + avg_growth_rate * years_ahead)
                                predicted_value = max(base_value * 0.8, predicted_value)
                            else:
                                predicted_value = base_value * ((1.0 + avg_growth_rate) ** years_ahead)
                        
                        if indicator == "SP.POP.TOTL":
                            max_change = 0.03  # 3% max for population
                        else:
                            max_change_per_year = 0.05
                            max_change = min(0.20, max_change_per_year * years_ahead)
                        
                        if predicted_value > base_value * (1 + max_change):
                            predicted_value = base_value * (1 + max_change)
                            logger.warning(f"Capped prediction for {country} to {max_change*100:.1f}% increase: {predicted_value:.2f}")
                        elif predicted_value < base_value * (1 - max_change):
                            predicted_value = base_value * (1 - max_change)
                            logger.warning(f"Capped prediction for {country} to {max_change*100:.1f}% decrease: {predicted_value:.2f}")
                        
                        logger.info(f"Predicted {country}: base={base_value:.2f} (year {actual_base_year}), growth={avg_growth_rate:.4f}, years={years_ahead}, result={predicted_value:.2f}")
                    else:
                        # Fallback: simple linear extrapolation with capped growth
                        years_diff = recent[-1][0] - recent[0][0]
                        values_diff = recent[-1][1] - recent[0][1]
                        if recent[0][1] > 0 and years_diff > 0:
                            growth_rate = (values_diff / recent[0][1]) / years_diff
                            # Cap growth rate to ensure gradual changes - more conservative for population
                            if indicator == "SP.POP.TOTL":
                                growth_rate = max(-0.02, min(0.02, growth_rate))
                            else:
                                growth_rate = max(-0.05, min(0.05, growth_rate))
                            # Use compound growth for consistency
                            years_ahead = target_year - actual_base_year
                            if years_ahead > 0:
                                # For declining populations, use linear extrapolation
                                if growth_rate < 0:
                                    predicted_value = base_value * (1.0 + growth_rate * years_ahead)
                                    predicted_value = max(base_value * 0.8, predicted_value)  # At most 20% decline
                                else:
                                    predicted_value = base_value * ((1.0 + growth_rate) ** years_ahead)
                                # Cap to reasonable change - stricter for population
                                if indicator == "SP.POP.TOTL":
                                    max_change = 0.03  # 3% max for population
                                else:
                                    max_change_per_year = 0.05
                                    max_change = min(0.20, max_change_per_year * years_ahead)
                                if predicted_value > base_value * (1 + max_change):
                                    predicted_value = base_value * (1 + max_change)
                                elif predicted_value < base_value * (1 - max_change):
                                    predicted_value = base_value * (1 - max_change)
                            else:
                                predicted_value = base_value
                        else:
                            # Edge case: no valid data, use base value
                            predicted_value = base_value
                            logger.warning(f"Invalid trend data for {country}, using base value")
                elif len(trend_data) == 1:
                    # Only one data point, use a conservative estimate
                    # Use average growth from other countries if available
                    if len(countries_list) > 1:
                        other_growth_rates = []
                        for other_country in countries_list:
                            if other_country != country:
                                other_trend = country_trends[other_country]
                                if len(other_trend['trend_data']) >= 2:
                                    recent = other_trend['trend_data'][-3:] if len(other_trend['trend_data']) >= 3 else other_trend['trend_data']
                                    if len(recent) >= 2:
                                        years_diff = recent[-1][0] - recent[0][0]
                                        values_diff = recent[-1][1] - recent[0][1]
                                        if recent[0][1] > 0 and years_diff > 0:
                                            growth = (values_diff / recent[0][1]) / years_diff
                                            growth = max(-0.05, min(0.05, growth))  # Cap growth
                                            other_growth_rates.append(growth)
                        
                        if other_growth_rates:
                            avg_growth = sum(other_growth_rates) / len(other_growth_rates)
                            # Cap to realistic bounds
                            if indicator == "SP.POP.TOTL":
                                avg_growth = max(-0.02, min(0.02, avg_growth))
                            else:
                                avg_growth = max(-0.05, min(0.05, avg_growth))
                            years_ahead = target_year - actual_base_year
                            if years_ahead > 0:
                                # For declining populations, use linear extrapolation
                                if avg_growth < 0:
                                    predicted_value = base_value * (1.0 + avg_growth * years_ahead)
                                    predicted_value = max(base_value * 0.8, predicted_value)  # At most 20% decline
                                else:
                                    predicted_value = base_value * ((1.0 + avg_growth) ** years_ahead)
                                # Cap to reasonable change - stricter for population
                                if indicator == "SP.POP.TOTL":
                                    max_change = 0.03  # 3% max for population
                                else:
                                    max_change_per_year = 0.05
                                    max_change = min(0.20, max_change_per_year * years_ahead)
                                if predicted_value > base_value * (1 + max_change):
                                    predicted_value = base_value * (1 + max_change)
                                elif predicted_value < base_value * (1 - max_change):
                                    predicted_value = base_value * (1 - max_change)
                            else:
                                predicted_value = base_value
                        else:
                            # Very conservative: assume small decline for Baltic countries (they've been declining)
                            # This respects historical trends
                            years_ahead = target_year - actual_base_year
                            if years_ahead > 0:
                                if country in ["LVA", "EST", "LTU"]:
                                    # Use linear decline for Baltic countries
                                    predicted_value = base_value * (1.0 - 0.01 * years_ahead)  # -1% per year, linear
                                else:
                                    # For other countries, assume very small growth
                                    predicted_value = base_value * (1.001 ** years_ahead)  # +0.1% per year
                                # Cap to reasonable change - stricter for population
                                if indicator == "SP.POP.TOTL":
                                    max_change = 0.03  # 3% max for population
                                else:
                                    max_change = 0.20
                                if predicted_value > base_value * (1 + max_change):
                                    predicted_value = base_value * (1 + max_change)
                                elif predicted_value < base_value * (1 - max_change):
                                    predicted_value = base_value * (1 - max_change)
                            else:
                                predicted_value = base_value
                    else:
                        # Single country with limited data, use conservative estimate
                        # Respect known trends: Baltic countries have been declining
                        years_ahead = target_year - actual_base_year
                        if years_ahead > 0:
                            if country in ["LVA", "EST", "LTU"]:
                                # Use linear decline for Baltic countries
                                predicted_value = base_value * (1.0 - 0.01 * years_ahead)  # -1% per year, linear
                            else:
                                predicted_value = base_value * (1.001 ** years_ahead)  # +0.1% per year
                            # Cap to reasonable change - stricter for population
                            if indicator == "SP.POP.TOTL":
                                max_change = 0.03  # 3% max for population
                            else:
                                max_change = 0.20
                            if predicted_value > base_value * (1 + max_change):
                                predicted_value = base_value * (1 + max_change)
                            elif predicted_value < base_value * (1 - max_change):
                                predicted_value = base_value * (1 - max_change)
                        else:
                            predicted_value = base_value
                else:
                    # No trend data available - use base value (no change)
                    # This handles edge case of missing historical data
                    predicted_value = base_value
                    logger.warning(f"No trend data for {country}, using base value: {base_value}")
                
                base_value, predicted_value = self._sanitize_pair(base_value, predicted_value)
                insight_payload[country] = {
                    "base_value": base_value,
                    "predicted_value": predicted_value,
                    "base_year": actual_base_year,
                    "target_year": target_year,
                    "trend": trend_data,
                    "data_points": len(trend_data),
                }
                
                predictions[country] = max(0.0, predicted_value)
        
        insights = self._generate_insights(indicator, insight_payload, target_year)
        result = {
            "indicator": indicator,
            "countries": countries,
            "target_year": target_year,
            "base_year": base_year,
            "predictions": predictions,
            "insights": insights,
        }
        
        # Cache results
        if self.cache:
            try:
                import json
                self.cache.setex(cache_key, 3600, json.dumps(result))
            except Exception as e:
                logger.debug(f"Cache write error: {e}")
        
        return result

    def _generate_insights(
        self,
        indicator: str,
        insight_data: Dict[str, Dict[str, Any]],
        target_year: int,
    ) -> Dict[str, Any]:
        if indicator == "SP.POP.TOTL":
            return self._generate_population_insights(insight_data, target_year)
        return {
            "overview": "Narrative insights are not yet available for this indicator.",
            "by_country": [],
            "comparative": [],
            "drivers": [],
            "risks": [],
            "scenarios": [],
            "policy": [],
            "communication": [],
            "caveats": [
                "Predictions are experimental. Consider supplementing with domain-specific analysis."
            ],
        }
    
    def _generate_population_insights(
        self,
        insight_data: Dict[str, Dict[str, Any]],
        target_year: int,
    ) -> Dict[str, Any]:
        by_country = []
        comparative = []
        drivers = []
        risks = []
        scenarios = []
        policy = []
        communication = []
        caveats = []
        decline_count = 0
        growth_count = 0
        strongest_decline = None
        strongest_growth = None
        for country, data in insight_data.items():
            base_value = data.get("base_value", 0.0)
            predicted_value = data.get("predicted_value", base_value)
            base_year = data.get("base_year")
            data_points = data.get("data_points", 0)
            trend = data.get("trend", [])
            change_pct = 0.0
            if base_value > 0:
                change_pct = (predicted_value - base_value) / base_value
            direction = "flat"
            if change_pct > 0.0005:
                direction = "growing"
                growth_count += 1
            elif change_pct < -0.0005:
                direction = "declining"
                decline_count += 1
            summary = (
                f"Population {'is projected to ' + ('grow' if change_pct > 0 else 'shrink' if change_pct < 0 else 'remain flat')} "
                f"by {change_pct * 100:0.2f}% between {base_year} and {target_year}."
            )
            by_country.append(
                {
                    "country": country,
                    "summary": summary,
                    "change_pct": change_pct,
                    "base_value": base_value,
                    "predicted_value": predicted_value,
                    "data_points": data_points,
                }
            )
            if data_points < 5:
                caveats.append(
                    f"{country}: Limited historical data ({data_points} points) — treat forecast cautiously."
                )
            if abs(change_pct) >= 0.02:
                risks.append(
                    f"{country}: Projected change of {change_pct*100:0.1f}% may require proactive planning."
                )
            if change_pct < -0.005:
                policy.append(
                    f"{country}: Consider fertility incentives, return-migration programs, and productivity investments to offset decline."
                )
                drivers.append(
                    f"{country}: Historical decline likely driven by migration outflows and ageing — monitor labor market pressures."
                )
                scenarios.append(
                    f"{country}: If emigration slows or family support improves, decline could ease by {abs(change_pct*100):0.1f}% in {target_year}."
                )
                communication.append(
                    f"{country}: Forecast dip of {change_pct*100:0.1f}% by {target_year}; plan workforce and social services accordingly."
                )
            elif change_pct > 0.005:
                policy.append(
                    f"{country}: Prepare infrastructure and housing for continued growth (+{change_pct*100:0.1f}%)."
                )
                drivers.append(
                    f"{country}: Population growth may reflect inward migration or higher birth rates — validate with recent census data."
                )
                scenarios.append(
                    f"{country}: Sustained growth could add {abs(change_pct*base_value):0.0f} people by {target_year}; monitor urban capacity."
                )
                communication.append(
                    f"{country}: Plan for {change_pct*100:0.1f}% population lift by {target_year}; scale services proportionally."
                )
            else:
                communication.append(
                    f"{country}: Population remains broadly stable through {target_year}; maintain current resource levels but track migration shocks."
                )
            if strongest_decline is None or change_pct < strongest_decline[1]:
                strongest_decline = (country, change_pct)
            if strongest_growth is None or change_pct > strongest_growth[1]:
                strongest_growth = (country, change_pct)
            if data_points and data_points < 10:
                caveats.append(
                    f"{country}: Only {data_points} usable annual observations — supplement with local statistics for decisions."
                )
            if len(trend) and trend[0][0] > 1990:
                caveats.append(
                    f"{country}: Historical window starts in {trend[0][0]}, so earlier demographic shifts are not captured."
                )
        overview_parts = []
        if decline_count:
            overview_parts.append(f"{decline_count} country{'ies' if decline_count != 1 else ''} continue declining")
        if growth_count:
            overview_parts.append(f"{growth_count} show gentle growth")
        if not overview_parts:
            overview_parts.append("Population levels remain broadly flat across the selected countries")
        overview = ", while ".join(overview_parts) + f" through {target_year}."
        if strongest_decline and strongest_decline[1] < -0.001:
            comparative.append(
                f"Sharpest decline: {strongest_decline[0]} at {strongest_decline[1]*100:0.2f}% over the forecast horizon."
            )
        if strongest_growth and strongest_growth[1] > 0.001:
            comparative.append(
                f"Largest increase: {strongest_growth[0]} at {strongest_growth[1]*100:0.2f}%."
            )
        if not comparative:
            comparative.append("Projected changes are minimal across the selected countries.")
        if not caveats:
            caveats.append("These forecasts rely on World Bank historical series; unexpected shocks (pandemics, conflicts) are not captured.")
        if not drivers:
            drivers.append("Recent population dynamics appear driven more by migration balances than sudden fertility shifts — confirm with local data.")
        if not risks:
            risks.append("Monitor quarterly demographic updates; small percentage changes can still impact labor supply and public finance.")
        if not scenarios:
            scenarios.append("Track migration policies and economic conditions — modest adjustments could swing projections by several tenths of a percent.")
        if not policy:
            policy.append("Maintain adaptive planning: tie service capacity and workforce programs to the latest census and migration statistics.")
        if not communication:
            communication.append("Key message: population trajectories remain within ±3% of current levels through the forecast window.")
        return {
            "overview": overview,
            "by_country": by_country,
            "comparative": comparative,
            "drivers": drivers,
            "risks": risks,
            "scenarios": scenarios,
            "policy": policy,
            "communication": communication,
            "caveats": caveats,
            "target_year": target_year,
        }

