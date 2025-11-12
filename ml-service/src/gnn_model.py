"""
Graph Neural Network model for predicting migration flows between countries.
Models countries as nodes and migration as edges.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np


class MigrationGNN(nn.Module):
    """
    Graph Neural Network for predicting migration flows.
    
    Architecture:
    - Nodes: Countries with features (GDP, population, etc.)
    - Edges: Migration flows between countries
    - Output: Predicted edge weights (future migration volumes)
    """
    
    def __init__(
        self,
        node_features: int = 10,  # GDP, population, life expectancy, etc.
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_gat: bool = True,  # Use Graph Attention Network instead of GCN
    ):
        super(MigrationGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gat = use_gat
        
        # Input projection
        if use_gat:
            self.conv1 = GATConv(node_features, hidden_dim, heads=4, dropout=dropout, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=2, dropout=dropout, concat=True)
            self.conv3 = GATConv(hidden_dim * 2, hidden_dim, heads=1, dropout=dropout, concat=False)
        else:
            self.conv1 = GCNConv(node_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        # Use LayerNorm instead of BatchNorm to avoid issues with single node graphs
        self.bn1 = nn.LayerNorm(hidden_dim * 4 if use_gat else hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim * 2 if use_gat else hidden_dim)
        
        # Edge prediction head
        # Predict edge weights from node embeddings
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single value: migration volume
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge attributes (optional)
        
        Returns:
            Predicted edge weights [num_edges, 1]
        """
        x, edge_index = data.x, data.edge_index
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Predict edge weights from node embeddings
        # For each edge, concatenate source and target node embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=1)
        
        # Predict migration volume for each edge
        edge_weights = self.edge_predictor(edge_embeddings)
        
        return edge_weights.squeeze(-1)  # [num_edges]
    
    def predict_migration(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        device: str = "cpu"
    ) -> np.ndarray:
        """
        Predict migration flows for given graph.
        
        Args:
            node_features: Node feature matrix [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            device: Device to run inference on
        
        Returns:
            Predicted migration volumes [num_edges]
        """
        self.eval()
        
        # Convert to tensors
        x = torch.FloatTensor(node_features).to(device)
        edge_idx = torch.LongTensor(edge_index).to(device)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_idx)
        
        with torch.no_grad():
            predictions = self.forward(data)
        
        return predictions.cpu().numpy()


class CountryGraphBuilder:
    """
    Builds graph data structures from country and migration data.
    """
    
    def __init__(self):
        self.country_to_idx: Dict[str, int] = {}
        self.idx_to_country: Dict[int, str] = {}
    
    def build_graph(
        self,
        country_features: Dict[str, np.ndarray],  # country_code -> feature vector
        migration_flows: List[Tuple[str, str, float]],  # (source, target, volume)
    ) -> Data:
        """
        Build a PyTorch Geometric Data object from country and migration data.
        
        Args:
            country_features: Dictionary mapping country codes to feature vectors
            migration_flows: List of (source_country, target_country, migration_volume) tuples
        
        Returns:
            PyTorch Geometric Data object
        """
        # Create country index mapping
        all_countries = set(country_features.keys())
        for src, tgt, _ in migration_flows:
            all_countries.add(src)
            all_countries.add(tgt)
        
        countries = sorted(list(all_countries))
        self.country_to_idx = {country: idx for idx, country in enumerate(countries)}
        self.idx_to_country = {idx: country for country, idx in self.country_to_idx.items()}
        
        # Build node features matrix
        num_nodes = len(countries)
        num_features = len(next(iter(country_features.values())))
        node_features = np.zeros((num_nodes, num_features))
        
        for country, idx in self.country_to_idx.items():
            if country in country_features:
                node_features[idx] = country_features[country]
            else:
                # Use zero vector for countries without features
                node_features[idx] = np.zeros(num_features)
        
        # Build edge index and edge attributes
        edge_list = []
        edge_weights = []
        
        for src, tgt, volume in migration_flows:
            if src in self.country_to_idx and tgt in self.country_to_idx:
                src_idx = self.country_to_idx[src]
                tgt_idx = self.country_to_idx[tgt]
                edge_list.append([src_idx, tgt_idx])
                edge_weights.append(volume)
        
        if len(edge_list) == 0:
            # Create empty graph
            edge_index = np.array([[], []], dtype=np.int64)
        else:
            edge_index = np.array(edge_list).T
        
        # Convert to tensors
        x = torch.FloatTensor(node_features)
        edge_idx = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_weights) if edge_weights else None
        
        return Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
    
    def get_country_index(self, country_code: str) -> Optional[int]:
        """Get the node index for a country code."""
        return self.country_to_idx.get(country_code)
    
    def get_country_code(self, node_index: int) -> Optional[str]:
        """Get the country code for a node index."""
        return self.idx_to_country.get(node_index)

