# ML Prediction Service

Graph Neural Network (GNN) service for predicting population growth and migration flows.

## Architecture

- **Nodes**: Countries with features (GDP, population, life expectancy, etc.)
- **Edges**: Migration flows between countries
- **Model**: Graph Attention Network (GAT) using PyTorch Geometric
- **Caching**: Redis/ElastiCache for prediction results

## Features

1. **Population Prediction**: Predict future population for a country
2. **Migration Flow Prediction**: Predict migration volumes between countries

## Setup

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python -m uvicorn src.api:app --host 0.0.0.0 --port 5000
```

### Docker

```bash
docker-compose up ml-service
```

## API Endpoints

### Health Check
```
GET /health
```

### Population Prediction
```
GET /api/predict/population?country=USA&years_ahead=5&base_year=2020
POST /api/predict/population
{
  "country": "USA",
  "years_ahead": 5,
  "base_year": 2020
}
```

### Migration Flow Prediction
```
POST /api/predict/migration
{
  "countries": ["USA", "MEX", "CAN"],
  "target_year": 2025,
  "base_year": 2020
}
```

## Model Training

The GNN model needs to be trained on historical data. To train:

1. Collect historical migration data
2. Prepare country features for multiple years
3. Train the model using PyTorch
4. Save model weights to `models/migration_gnn.pth`

Example training script (to be implemented):
```python
from src.gnn_model import MigrationGNN, CountryGraphBuilder
from src.data_loader import WorldBankDataLoader
import torch

# Load data
loader = WorldBankDataLoader()
# ... prepare training data ...

# Train model
model = MigrationGNN()
# ... training loop ...

# Save model
torch.save(model.state_dict(), "models/migration_gnn.pth")
```

## AWS ElastiCache Configuration

To use AWS ElastiCache instead of local Redis:

1. Create an ElastiCache Redis cluster in AWS
2. Update `docker-compose.yml`:
```yaml
cache:
  # Remove local Redis service
  # Use environment variables instead:
  environment:
    REDIS_HOST: <your-elasticache-endpoint>
    REDIS_PORT: 6379
```

3. Update security groups to allow connection from your services

## Data Sources

- **Country Features**: World Bank API
- **Migration Data**: Currently placeholder - needs integration with:
  - UN Migration Data Portal
  - World Bank Migration Data
  - Other migration databases

## Future Improvements

1. Add actual migration data source integration
2. Implement temporal GNN for time-series predictions
3. Add model training pipeline
4. Add model versioning and A/B testing
5. Add prediction confidence intervals
6. Extend to all World Bank indicators

