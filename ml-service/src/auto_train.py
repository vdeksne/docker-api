"""
Automatic training script that collects data from World Bank API and trains the model.
This can be run on startup or on a schedule.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import os

from .gnn_model import MigrationGNN
from .data_loader import WorldBankDataLoader
from .migration_data import MigrationDataFetcher
from .train import GNNTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_training_data(
    data_loader: WorldBankDataLoader,
    migration_fetcher: MigrationDataFetcher,
    countries: List[str],
    years: List[int],
    indicator: str = "SP.POP.TOTL",
) -> Tuple[List, List]:
    """
    Collect training data from World Bank API.
    
    Args:
        data_loader: World Bank data loader
        migration_fetcher: Migration data fetcher
        countries: List of country codes
        years: List of years
        indicator: Indicator to train on
    
    Returns:
        Tuple of (graphs, targets) for training
    """
    from torch_geometric.data import Data
    
    graphs = []
    targets = []
    graph_builder = CountryGraphBuilder()
    
    logger.info(f"Collecting training data for {len(countries)} countries, {len(years)} years")
    
    for year_idx, year in enumerate(years):
        if year_idx % 5 == 0:
            logger.info(f"Processing year {year} ({year_idx + 1}/{len(years)})")
        
        try:
            # Get country features for this year
            country_features = {}
            for country in countries:
                try:
                    features = data_loader.get_country_features(country, year)
                    country_features[country] = features
                except Exception as e:
                    logger.debug(f"Error fetching features for {country} year {year}: {e}")
                    continue
            
            if len(country_features) < 2:
                logger.debug(f"Skipping year {year}: not enough countries with data")
                continue
            
            # Get migration flows for this year
            migration_flows = migration_fetcher.estimate_migration_flows(
                list(country_features.keys()), year
            )
            
            # If no migration flows, create synthetic edges based on indicator values
            if len(migration_flows) == 0:
                # Get indicator values for this year
                indicator_values = {}
                for country in country_features.keys():
                    try:
                        value = data_loader.fetch_indicator_data(country, indicator, year, year)
                        indicator_values[country] = value or 0.0
                    except:
                        indicator_values[country] = 0.0
                
                # Create edges based on indicator similarity and GDP differences
                countries_list = list(country_features.keys())
                for i, src in enumerate(countries_list):
                    for j, tgt in enumerate(countries_list):
                        if i != j:
                            src_val = indicator_values.get(src, 0.0)
                            tgt_val = indicator_values.get(tgt, 0.0)
                            
                            # Similar values = stronger connection
                            if max(abs(src_val), abs(tgt_val), 1.0) > 0:
                                similarity = 1.0 / (1.0 + abs(src_val - tgt_val) / max(abs(src_val), abs(tgt_val), 1.0))
                                # Scale to reasonable migration volume
                                flow = similarity * abs(src_val) * 0.001  # Scale factor
                                migration_flows.append((src, tgt, flow))
            
            if len(migration_flows) == 0:
                continue
            
            # Build graph
            graph_data = graph_builder.build_graph(country_features, migration_flows)
            
            # Extract target edge weights
            edge_weights = graph_data.edge_attr
            if edge_weights is None or len(edge_weights) == 0:
                continue
            
            graphs.append(graph_data)
            targets.append(edge_weights)
            
        except Exception as e:
            logger.warning(f"Error processing year {year}: {e}")
            continue
    
    logger.info(f"Collected {len(graphs)} training samples")
    return graphs, targets


def auto_train(
    model_path: str = "models/migration_gnn.pth",
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    epochs: int = 50,
    cache: Optional[object] = None,
) -> bool:
    """
    Automatically train the model with data from World Bank API.
    
    Args:
        model_path: Path to save trained model
        countries: List of countries (defaults to common countries)
        years: List of years (defaults to 2000-2020)
        epochs: Number of training epochs
        cache: Redis cache client
    
    Returns:
        True if training succeeded
    """
    # Default countries if not provided
    # Prioritize Baltic countries (LVA, EST, LTU) for better predictions
    if countries is None:
        countries = [
            # Baltic countries first (most important for this use case)
            "LVA", "EST", "LTU",
            # Neighboring countries for context
            "POL", "SWE", "FIN", "DEU",
            # Additional countries for broader training
            "USA", "MEX", "CAN", "GBR", "FRA", "ITA", "ESP",
            "CHN", "IND", "JPN", "KOR", "BRA", "ARG", "AUS", "NZL",
            "CZE", "HUN", "ROU", "BGR"
        ]
    
    # Default years if not provided
    # Use more recent years and extend to 2023 if available for better predictions
    if years is None:
        years = list(range(1990, 2024))  # 1990-2023 for more data
    
    logger.info(f"Starting automatic training with {len(countries)} countries, {len(years)} years")
    
    # Initialize components
    data_loader = WorldBankDataLoader(cache=cache)
    migration_fetcher = MigrationDataFetcher(cache=cache)
    
    # Initialize model
    model = MigrationGNN(
        node_features=10,
        hidden_dim=64,
        num_layers=3,
        dropout=0.2,
        use_gat=True,
    )
    
    # Use trainer's efficient data collection method
    from .train import GNNTrainer
    trainer = GNNTrainer(model, device="cpu", cache=cache)
    
    # Collect training data
    logger.info("Collecting training data from World Bank API...")
    graphs, targets = trainer.prepare_training_data_from_api(countries, years)
    
    if len(graphs) == 0:
        logger.error("No training data collected. Cannot train model.")
        return False
    
    # Train model
    logger.info(f"Training model with {len(graphs)} samples for {epochs} epochs...")
    try:
        train_losses, val_losses = trainer.train(
            graphs, targets, epochs=epochs, learning_rate=0.001
        )
        
        # Save model
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_path)
        
        logger.info(f"Training complete! Model saved to {model_path}")
        logger.info(f"Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"Final val loss: {val_losses[-1]:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-train GNN model")
    parser.add_argument("--model-path", type=str, default="models/migration_gnn.pth")
    parser.add_argument("--countries", nargs="+", help="Country codes")
    parser.add_argument("--years", nargs="+", type=int, help="Years")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cache", action="store_true", help="Use Redis cache")
    
    args = parser.parse_args()
    
    cache = None
    if args.cache:
        try:
            import redis
            import os
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD", None)
            redis_kwargs = {"host": redis_host, "port": redis_port, "decode_responses": True}
            if redis_password:
                redis_kwargs["password"] = redis_password
            cache = redis.Redis(**redis_kwargs)
            cache.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
    
    success = auto_train(
        model_path=args.model_path,
        countries=args.countries,
        years=args.years,
        epochs=args.epochs,
        cache=cache,
    )
    
    exit(0 if success else 1)

