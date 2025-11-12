"""
Prediction service for migration flows using GNN.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

from .gnn_model import MigrationGNN, CountryGraphBuilder
from .data_loader import WorldBankDataLoader
from .migration_data import MigrationDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationPredictor:
    """Service for predicting migration flows using GNN."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        cache: Optional[object] = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model weights (optional)
            device: Device to run inference on ("cpu" or "cuda")
            cache: Redis cache client (optional)
        """
        self.device = device
        self.cache = cache
        
        # Initialize model
        self.model = MigrationGNN(
            node_features=10,  # Number of features from data_loader
            hidden_dim=64,
            num_layers=3,
            dropout=0.2,
            use_gat=True,
        ).to(device)
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.eval()
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
        else:
            logger.warning("No model weights found. Using untrained model.")
        
        # Initialize data loader, migration fetcher, and graph builder
        self.data_loader = WorldBankDataLoader(cache=cache)
        self.migration_fetcher = MigrationDataFetcher(cache=cache)
        self.graph_builder = CountryGraphBuilder()
    
    def predict_migration_flows(
        self,
        countries: List[str],
        target_year: Optional[int] = None,
        base_year: Optional[int] = None,
    ) -> Dict[Tuple[str, str], float]:
        """
        Predict migration flows between countries.
        
        Args:
            countries: List of country ISO3 codes
            target_year: Year to predict for (defaults to next year)
            base_year: Year to use as base for features (defaults to most recent)
        
        Returns:
            Dictionary mapping (source_country, target_country) tuples to predicted migration volumes
        """
        if target_year is None:
            target_year = 2025  # Default to next year
        
        if base_year is None:
            base_year = 2020  # Default base year
        
        # Check cache
        cache_key = f"migration_pred:{':'.join(sorted(countries))}:{target_year}"
        if self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Get country features
        logger.info(f"Fetching features for {len(countries)} countries")
        country_features = self.data_loader.get_country_features_batch(
            countries, base_year
        )
        
        # Build graph using estimated migration flows
        # Get estimated flows based on historical patterns
        migration_flows = self.migration_fetcher.estimate_migration_flows(
            countries, base_year
        )
        
        # If no flows estimated, create placeholder edges
        if len(migration_flows) == 0:
            logger.warning("No migration flows estimated, using placeholder edges")
            for i, src in enumerate(countries):
                for j, tgt in enumerate(countries):
                    if i != j:
                        migration_flows.append((src, tgt, 0.0))
        
        graph_data = self.graph_builder.build_graph(country_features, migration_flows)
        graph_data = graph_data.to(self.device)
        
        # Predict
        logger.info("Running GNN prediction")
        with torch.no_grad():
            predictions = self.model(graph_data)
        
        # Map predictions back to country pairs
        predictions_dict = {}
        edge_index = graph_data.edge_index.cpu().numpy()
        
        for idx, (src_idx, tgt_idx) in enumerate(edge_index.T):
            src_country = self.graph_builder.get_country_code(int(src_idx))
            tgt_country = self.graph_builder.get_country_code(int(tgt_idx))
            
            if src_country and tgt_country:
                pred_value = float(predictions[idx].cpu().numpy())
                predictions_dict[(src_country, tgt_country)] = max(0, pred_value)  # Ensure non-negative
        
        # Cache results
        if self.cache:
            try:
                self.cache.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(predictions_dict)
                )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        return predictions_dict
    
    def predict_population_growth(
        self,
        country: str,
        years_ahead: int = 5,
        base_year: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Predict population growth for a country.
        
        This is a simplified version that uses the GNN to predict
        net migration, then applies it to population growth.
        
        Args:
            country: Country ISO3 code
            years_ahead: Number of years to predict ahead
            base_year: Base year for prediction
        
        Returns:
            Dictionary mapping year to predicted population
        """
        if base_year is None:
            base_year = 2020
        
        # Get current population
        current_pop = self.data_loader.fetch_indicator_data(
            country, "SP.POP.TOTL", base_year, base_year
        )
        
        if current_pop is None:
            logger.warning(f"No population data for {country}")
            return {}
        
        # For now, use simple linear extrapolation
        # In production, this would use the GNN to predict migration flows
        # and combine with birth/death rates
        
        predictions = {}
        growth_rate = 0.01  # Placeholder: 1% annual growth
        
        for i in range(1, years_ahead + 1):
            year = base_year + i
            predicted_pop = current_pop * ((1 + growth_rate) ** i)
            predictions[year] = predicted_pop
        
        return predictions

