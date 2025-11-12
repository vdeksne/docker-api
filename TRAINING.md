# Automatic Model Training

The ML service now automatically trains the model on startup using real data from the World Bank API.

## How It Works

1. **Automatic Training on Startup**: When the ML service starts, it checks if a trained model exists. If not (or if `AUTO_TRAIN=true`), it automatically:
   - Collects training data from World Bank API for 24 countries over 21 years (2000-2020)
   - Trains the GNN model for 30 epochs
   - Saves the trained model to `/app/models/migration_gnn.pth`

2. **Data Collection**: The training script:
   - Fetches country features (GDP, population, life expectancy, etc.) from World Bank API
   - Estimates migration flows between countries based on:
     - Net migration rates from World Bank
     - GDP per capita differences
     - Population sizes
   - Creates synthetic edges when migration data is unavailable

3. **Model Training**: Uses the collected data to train a Graph Neural Network that:
   - Models countries as nodes
   - Models migration/relationships as edges
   - Learns to predict future indicator values based on historical patterns

## Configuration

In `docker-compose.yml`:
```yaml
ml-service:
  environment:
    AUTO_TRAIN: "true"  # Enable automatic training
    TRAINING_EPOCHS: "30"  # Number of training epochs
    MODEL_PATH: /app/models/migration_gnn.pth
  volumes:
    - ml_models:/app/models  # Persist trained models
```

## Manual Training

You can also trigger training manually via API:

```bash
POST http://localhost:5001/api/train
Content-Type: application/json

{
  "countries": ["USA", "MEX", "CAN"],
  "years": [2010, 2011, 2012, 2013, 2014],
  "epochs": 50
}
```

## Training Progress

The training process:
1. Collects data from World Bank API (this can take 5-10 minutes for 24 countries × 21 years)
2. Prepares graph data structures
3. Trains the model for the specified number of epochs
4. Saves the trained model

You can monitor progress in the logs:
```bash
docker-compose logs -f ml-service
```

## Notes

- The first training run will take longer as it fetches all data from the API
- Subsequent training runs will be faster due to Redis caching
- The trained model is persisted in a Docker volume, so it survives container restarts
- If training fails, the service will continue with an untrained model (with warnings)

