# ML Prediction Service Setup Guide

## Overview

This project now includes a Graph Neural Network (GNN) service for predicting:
- **Population growth** for countries
- **Migration flows** between countries

The GNN models countries as nodes (with features like GDP, population, etc.) and migration as edges (with weights representing migration volumes).

## Architecture

```
┌─────────────┐
│  Frontend   │
│  (React)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│  Backend    │─────▶│  ML Service  │
│  (Express)  │      │  (FastAPI)   │
└──────┬──────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
┌─────────────┐      ┌──────────────┐
│ PostgreSQL  │      │ Redis Cache  │
│   Database  │      │ ElastiCache  │
└─────────────┘      └──────────────┘
```

## Components

### 1. ML Service (`ml-service/`)
- **Framework**: PyTorch Geometric (GNN)
- **Model**: Graph Attention Network (GAT)
- **API**: FastAPI
- **Port**: 5000

### 2. Backend Integration
- New route: `/api/predictions/population`
- New route: `/api/predictions/migration`
- Proxies requests to ML service

### 3. Frontend API
- `predictPopulation()` - Predict population growth
- `predictMigration()` - Predict migration flows

## Quick Start

### 1. Start All Services

```bash
docker-compose up --build
```

This will start:
- PostgreSQL database
- Redis cache
- Backend API (port 4000)
- ML Service (port 5000)
- Frontend (port 5173)

### 2. Test Population Prediction

```bash
# Via backend proxy
curl "http://localhost:4000/api/predictions/population?country=USA&years_ahead=5&base_year=2020"

# Direct ML service
curl "http://localhost:5000/api/predict/population?country=USA&years_ahead=5&base_year=2020"
```

### 3. Test Migration Prediction

```bash
curl -X POST "http://localhost:4000/api/predictions/migration" \
  -H "Content-Type: application/json" \
  -d '{
    "countries": ["USA", "MEX", "CAN"],
    "target_year": 2025,
    "base_year": 2020
  }'
```

## AWS ElastiCache Setup

### Option 1: Use Local Redis (Current)
The `docker-compose.yml` includes a local Redis instance.

### Option 2: Use AWS ElastiCache

1. **Create ElastiCache Cluster**:
   - Go to AWS Console → ElastiCache
   - Create Redis cluster
   - Note the endpoint URL

2. **Update `docker-compose.yml`**:
```yaml
cache:
  # Remove the image and command
  # Add environment variables:
  environment:
    REDIS_HOST: your-elasticache-endpoint.cache.amazonaws.com
    REDIS_PORT: 6379
```

3. **Update Security Groups**:
   - Allow inbound traffic on port 6379 from your services

4. **Update ML Service Environment**:
```yaml
ml-service:
  environment:
    REDIS_HOST: your-elasticache-endpoint.cache.amazonaws.com
    REDIS_PORT: 6379
    REDIS_PASSWORD: your-password-if-set
```

## Model Training

The current model is untrained (uses random weights). To train:

1. **Collect Training Data**:
   - Historical migration flows
   - Country features for multiple years

2. **Train Model**:
```python
from src.gnn_model import MigrationGNN
from src.data_loader import WorldBankDataLoader
import torch
import torch.nn as nn

# Load data
loader = WorldBankDataLoader()
# ... prepare training data ...

# Initialize model
model = MigrationGNN()

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    # ... training code ...
    pass

# Save model
torch.save(model.state_dict(), "models/migration_gnn.pth")
```

3. **Update Docker Compose**:
```yaml
ml-service:
  volumes:
    - ./ml-service/models:/app/models
```

## API Endpoints

### Backend Proxy Endpoints

#### Population Prediction
```
GET /api/predictions/population?country=USA&years_ahead=5&base_year=2020
```

#### Migration Prediction
```
POST /api/predictions/migration
Body: {
  "countries": ["USA", "MEX"],
  "target_year": 2025,
  "base_year": 2020
}
```

### Direct ML Service Endpoints

#### Health Check
```
GET http://localhost:5000/health
```

#### Population Prediction
```
GET /api/predict/population?country=USA&years_ahead=5&base_year=2020
POST /api/predict/population
{
  "country": "USA",
  "years_ahead": 5,
  "base_year": 2020
}
```

#### Migration Prediction
```
POST /api/predict/migration
{
  "countries": ["USA", "MEX", "CAN"],
  "target_year": 2025,
  "base_year": 2020
}
```

## Frontend Integration

```typescript
import { predictPopulation, predictMigration } from "./services/api";

// Predict population
const result = await predictPopulation({
  country: "USA",
  years_ahead: 5,
  base_year: 2020,
});

console.log(result.predictions); // { 2021: 332000000, 2022: 335000000, ... }

// Predict migration
const migration = await predictMigration({
  countries: ["USA", "MEX", "CAN"],
  target_year: 2025,
  base_year: 2020,
});

console.log(migration.predictions); // { "USA_MEX": 150000, "MEX_USA": 200000, ... }
```

## Caching

- **Redis/ElastiCache**: Caches prediction results for 1 hour
- **Cache Key Format**: `migration_pred:{countries}:{year}`
- **TTL**: 3600 seconds (1 hour)

## Future Enhancements

1. **Actual Migration Data**: Integrate with UN Migration Data Portal
2. **Temporal GNN**: Add time-series modeling
3. **Model Training Pipeline**: Automated training on new data
4. **Confidence Intervals**: Add uncertainty estimates
5. **All Indicators**: Extend predictions to all World Bank indicators

## Troubleshooting

### ML Service Not Starting
- Check Python dependencies: `pip install -r requirements.txt`
- Check port 5000 is available
- Check Redis connection

### Predictions Returning Empty
- Model may be untrained (using random weights)
- Check country codes are valid ISO3 codes
- Check World Bank API is accessible

### Redis Connection Issues
- Verify Redis is running: `docker-compose ps cache`
- Check Redis password matches
- For ElastiCache: verify security groups allow connection

## Files Structure

```
ml-service/
├── src/
│   ├── api.py              # FastAPI service
│   ├── gnn_model.py        # GNN model definition
│   ├── data_loader.py      # World Bank data fetching
│   └── predictor.py        # Prediction service
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # ML service documentation

backend/
└── src/
    └── routes/
        └── predictions.ts  # Prediction routes

frontend/
└── src/
    └── services/
        └── api.ts         # Prediction API functions
```

