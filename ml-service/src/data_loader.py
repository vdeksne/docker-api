"""
Data loading and preprocessing for migration prediction.
Fetches country features and migration data from World Bank API.
"""
import requests
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorldBankDataLoader:
    """Loads country features and migration data from World Bank API."""
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    # Indicator codes for country features
    FEATURE_INDICATORS = {
        "population": "SP.POP.TOTL",
        "gdp": "NY.GDP.MKTP.CD",
        "gdp_per_capita": "NY.GDP.PCAP.CD",
        "life_expectancy": "SP.DYN.LE00.IN",
        "inflation": "FP.CPI.TOTL.ZG",
        "trade": "NE.TRD.GNFS.ZS",
        "urban_population": "SP.URB.TOTL.IN.ZS",
        "health_expenditure": "SH.XPD.CHEX.GD.ZS",
        "education_expenditure": "SE.XPD.TOTL.GD.ZS",
        "internet_users": "IT.NET.USER.ZS",
    }
    
    def __init__(self, cache: Optional[object] = None):
        """
        Initialize data loader.
        
        Args:
            cache: Redis cache client (optional)
        """
        self.cache = cache
    
    def fetch_indicator_data(
        self,
        country: str,
        indicator: str,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
    ) -> Optional[float]:
        """
        Fetch a single indicator value for a country.
        
        Args:
            country: Country ISO3 code (e.g., "USA")
            indicator: Indicator code
            from_year: Start year (optional)
            to_year: End year (optional)
        
        Returns:
            Latest available value or None
        """
        # Check cache first
        cache_key = f"wb:{indicator}:{country}:{from_year or 'min'}:{to_year or 'max'}"
        if self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    data = json.loads(cached) if isinstance(cached, str) else cached
                    if data and len(data.get("rows", [])) > 0:
                        # Get most recent non-null value
                        rows = data.get("rows", [])
                        for row in reversed(rows):
                            if row.get("value") is not None:
                                return float(row["value"])
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Fetch from World Bank API
        url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}"
        params = {"format": "json", "per_page": 100}
        if from_year and to_year:
            params["date"] = f"{from_year}:{to_year}"
        elif from_year:
            params["date"] = str(from_year)
        elif to_year:
            params["date"] = str(to_year)
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list) or len(data) < 2:
                return None
            
            entries = data[1]
            if not entries:
                return None
            
            # Get most recent non-null value
            for entry in reversed(entries):
                if entry.get("value") is not None:
                    return float(entry["value"])
            
            return None
        except Exception as e:
            logger.error(f"Error fetching {indicator} for {country}: {e}")
            return None
    
    def get_country_features(
        self,
        country: str,
        year: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get feature vector for a country.
        
        Args:
            country: Country ISO3 code
            year: Year to fetch data for (defaults to most recent)
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        for feature_name, indicator_code in self.FEATURE_INDICATORS.items():
            value = self.fetch_indicator_data(country, indicator_code, year, year)
            
            # Normalize: use log for large values, handle None
            if value is None:
                features.append(0.0)
            elif feature_name in ["population", "gdp"]:
                # Log scale for large values
                features.append(np.log1p(value))
            else:
                features.append(value)
        
        return np.array(features, dtype=np.float32)
    
    def get_country_features_batch(
        self,
        countries: List[str],
        year: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get feature vectors for multiple countries.
        
        Args:
            countries: List of country ISO3 codes
            year: Year to fetch data for
        
        Returns:
            Dictionary mapping country codes to feature vectors
        """
        features_dict = {}
        
        for country in countries:
            features = self.get_country_features(country, year)
            features_dict[country] = features
        
        return features_dict
    
    def get_migration_flows(
        self,
        countries: List[str],
        year: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Get migration flows between countries.
        
        Note: World Bank doesn't have a direct migration flow API.
        This is a placeholder that would need to be replaced with actual
        migration data source (e.g., UN Migration Data, World Bank Migration Data).
        
        For now, we'll use a simple heuristic based on population differences
        and distance (would need actual migration data in production).
        
        Args:
            countries: List of country ISO3 codes
            year: Year to fetch data for
        
        Returns:
            List of (source_country, target_country, migration_volume) tuples
        """
        # TODO: Replace with actual migration data source
        # For now, return empty list - migration data would need to be
        # fetched from a different source (UN, World Bank Migration Data Portal, etc.)
        
        logger.warning(
            "Migration flows not yet implemented. "
            "Need to integrate with actual migration data source."
        )
        
        return []
    
    def prepare_training_data(
        self,
        countries: List[str],
        years: List[int],
    ) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, str, float]]]:
        """
        Prepare training data for multiple countries and years.
        
        Args:
            countries: List of country ISO3 codes
            years: List of years to fetch data for
        
        Returns:
            Tuple of (country_features_dict, migration_flows_list)
        """
        # Get features for all countries and years
        all_features = {}
        
        for year in years:
            year_features = self.get_country_features_batch(countries, year)
            for country, features in year_features.items():
                # Use year as key prefix to track temporal data
                all_features[f"{country}_{year}"] = features
        
        # Get migration flows (placeholder)
        migration_flows = self.get_migration_flows(countries, years[-1] if years else None)
        
        return all_features, migration_flows

