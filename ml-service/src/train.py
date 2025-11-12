"""
Training script for the GNN migration prediction model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from .gnn_model import MigrationGNN, CountryGraphBuilder
from .data_loader import WorldBankDataLoader
from .migration_data import MigrationDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNTrainer:
    """Trainer for the GNN migration prediction model."""
    
    def __init__(
        self,
        model: MigrationGNN,
        device: str = "cpu",
        cache: Optional[object] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.data_loader = WorldBankDataLoader(cache=cache)
        self.migration_fetcher = MigrationDataFetcher(cache=cache)
        self.graph_builder = CountryGraphBuilder()
    
    def prepare_training_data_from_api(
        self,
        countries: List[str],
        years: List[int],
        indicator: str = "SP.POP.TOTL",
    ) -> Tuple[List[Data], List[torch.Tensor]]:
        """
        Prepare training data by fetching from World Bank API.
        This is a more efficient version that collects data in batches.
        
        Args:
            countries: List of country ISO3 codes
            years: List of years to use for training
            indicator: Indicator to use for creating synthetic migration flows
        
        Returns:
            Tuple of (graph_data_list, target_edge_weights_list)
        """
        graphs = []
        targets = []
        
        logger.info(f"Preparing training data for {len(countries)} countries, {len(years)} years")
        
        # Prioritize Baltic countries - ensure they're always included
        baltic_countries = ["LVA", "EST", "LTU"]
        priority_countries = [c for c in countries if c in baltic_countries]
        other_countries = [c for c in countries if c not in baltic_countries]
        
        for year_idx, year in enumerate(years):
            if year_idx % 5 == 0:
                logger.info(f"Processing year {year} ({year_idx + 1}/{len(years)})")
            
            try:
                # Get country features for this year
                # Prioritize Baltic countries first
                country_features = {}
                
                # First, try to get Baltic countries (required)
                for country in priority_countries:
                    try:
                        features = self.data_loader.get_country_features(country, year)
                        country_features[country] = features
                    except Exception as e:
                        logger.warning(f"Error fetching features for {country} year {year}: {e}")
                        # Try to use cached or previous year's data
                        if year > min(years):
                            try:
                                features = self.data_loader.get_country_features(country, year - 1)
                                country_features[country] = features
                                logger.debug(f"Using previous year data for {country}")
                            except:
                                pass
                
                # Then get other countries
                for country in other_countries:
                    try:
                        features = self.data_loader.get_country_features(country, year)
                        country_features[country] = features
                    except Exception as e:
                        logger.debug(f"Error fetching features for {country} year {year}: {e}")
                        continue
                
                # Require at least Baltic countries to be present
                if len([c for c in priority_countries if c in country_features]) < 2:
                    logger.debug(f"Skipping year {year}: not enough Baltic countries with data")
                    continue
                
                if len(country_features) < 2:
                    continue
                
                # Get migration flows for this year
                migration_flows = self.migration_fetcher.estimate_migration_flows(
                    list(country_features.keys()), year
                )
                
                # If no migration flows, create synthetic edges based on indicator values
                if len(migration_flows) == 0:
                    # Get indicator values for this year
                    indicator_values = {}
                    for country in country_features.keys():
                        try:
                            value = self.data_loader.fetch_indicator_data(country, indicator, year, year)
                            indicator_values[country] = value or 0.0
                        except:
                            indicator_values[country] = 0.0
                    
                    # Create edges based on indicator similarity
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
                                    flow = similarity * abs(src_val) * 0.001
                                    migration_flows.append((src, tgt, flow))
                
                if len(migration_flows) == 0:
                    continue
                
                # Build graph
                graph_data = self.graph_builder.build_graph(country_features, migration_flows)
                
                # Extract target edge weights
                edge_weights = graph_data.edge_attr
                if edge_weights is None or len(edge_weights) == 0:
                    continue
                
                graphs.append(graph_data)
                targets.append(edge_weights)
                
            except Exception as e:
                logger.warning(f"Error processing year {year}: {e}")
                continue
        
        logger.info(f"Prepared {len(graphs)} training samples")
        return graphs, targets
    
    def prepare_training_data(
        self,
        countries: List[str],
        years: List[int],
    ) -> Tuple[List[Data], List[torch.Tensor]]:
        """
        Prepare training data from historical country and migration data.
        Uses the more efficient API-based data collection.
        
        Args:
            countries: List of country ISO3 codes
            years: List of years to use for training
        
        Returns:
            Tuple of (graph_data_list, target_edge_weights_list)
        """
        return self.prepare_training_data_from_api(countries, years)
    
    def train(
        self,
        graphs: List[Data],
        targets: List[torch.Tensor],
        epochs: int = 100,
        batch_size: int = 1,
        learning_rate: float = 0.001,
        train_split: float = 0.8,
    ) -> List[float]:
        """
        Train the GNN model.
        
        Args:
            graphs: List of graph data objects
            targets: List of target edge weight tensors
            epochs: Number of training epochs
            batch_size: Batch size (currently 1 due to variable graph sizes)
            learning_rate: Learning rate
            train_split: Fraction of data to use for training
        
        Returns:
            List of training losses per epoch
        """
        # Split data
        n_train = int(len(graphs) * train_split)
        train_graphs = graphs[:n_train]
        train_targets = targets[:n_train]
        val_graphs = graphs[n_train:]
        val_targets = targets[n_train:]
        
        logger.info(f"Training on {len(train_graphs)} samples, validating on {len(val_graphs)}")
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            
            for graph, target in zip(train_graphs, train_targets):
                graph = graph.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(graph)
                
                # Calculate loss
                loss = criterion(predictions, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_graphs)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for graph, target in zip(val_graphs, val_targets):
                    graph = graph.to(self.device)
                    target = target.to(self.device)
                    
                    predictions = self.model(graph)
                    loss = criterion(predictions, target)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_graphs) if val_graphs else 0.0
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )
        
        return train_losses, val_losses
    
    def save_model(self, path: str):
        """Save the trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GNN migration prediction model")
    parser.add_argument("--countries", nargs="+", default=["USA", "MEX", "CAN", "GBR", "FRA", "DEU", "CHN", "IND", "BRA", "ARG"],
                        help="List of country codes to train on")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2000, 2021)),
                        help="List of years to use for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/migration_gnn.pth",
                        help="Output path for trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu/cuda)")
    parser.add_argument("--cache", action="store_true", help="Use Redis cache")
    
    args = parser.parse_args()
    
    # Initialize model
    model = MigrationGNN(
        node_features=10,
        hidden_dim=64,
        num_layers=3,
        dropout=0.2,
        use_gat=True,
    )
    
    # Initialize cache if requested
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
    
    # Initialize trainer
    trainer = GNNTrainer(model, device=args.device, cache=cache)
    
    # Prepare training data
    logger.info("Preparing training data...")
    graphs, targets = trainer.prepare_training_data(args.countries, args.years)
    
    if len(graphs) == 0:
        logger.error("No training data prepared. Exiting.")
        return
    
    # Train model
    logger.info("Starting training...")
    train_losses, val_losses = trainer.train(
        graphs, targets, epochs=args.epochs, learning_rate=args.lr
    )
    
    # Save model
    trainer.save_model(args.output)
    
    # Save training history
    history_path = args.output.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epochs": args.epochs,
            "countries": args.countries,
            "years": args.years,
        }, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {args.output}")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"Final val loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()

