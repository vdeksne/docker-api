# Training and Testing Guide

## Overview

This guide covers:
1. Training the GNN model on historical migration data
2. Testing predictions
3. Using predictions for all World Bank indicators

## Training the Model

### Step 1: Prepare Training Data

The training script automatically fetches:
- Country features (GDP, population, etc.) from World Bank API
- Migration flows estimated from net migration rates and economic indicators

### Step 2: Run Training

```bash
# Inside the ml-service container or with Python environment
cd ml-service

# Basic training (10 countries, 2000-2020)
python -m src.train \
  --countries USA MEX CAN GBR FRA DEU CHN IND BRA ARG \
  --years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 \
  --epochs 100 \
  --lr 0.001 \
  --output models/migration_gnn.pth

# With more countries and years
python -m src.train \
  --countries USA MEX CAN GBR FRA DEU CHN IND BRA ARG JPN KOR AUS ITA ESP NLD BEL SWE NOR DNK \
  --years $(seq 2000 2020) \
  --epochs 200 \
  --lr 0.0005 \
  --output models/migration_gnn.pth \
  --cache  # Use Redis cache
```

### Step 3: Training Options

- `--countries`: List of country ISO3 codes
- `--years`: List of years for training data
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--output`: Path to save trained model (default: models/migration_gnn.pth)
- `--device`: Device to train on (cpu/cuda, default: cpu)
- `--cache`: Use Redis cache for data fetching

### Step 4: Monitor Training

The training script outputs:
- Training loss per epoch
- Validation loss per epoch
- Final model saved to specified path
- Training history saved as JSON

Example output:
```
Epoch 10/100 - Train Loss: 0.0234, Val Loss: 0.0256
Epoch 20/100 - Train Loss: 0.0189, Val Loss: 0.0212
...
Epoch 100/100 - Train Loss: 0.0123, Val Loss: 0.0145
Training complete. Model saved to models/migration_gnn.pth
Final train loss: 0.0123
Final val loss: 0.0145
```

## Testing Predictions

### Step 1: Start Services

```bash
# Start all services
docker-compose up --build

# Or start ML service separately
docker-compose up ml-service
```

### Step 2: Run Test Script

```bash
# Make test script executable
chmod +x ml-service/test_predictions.py

# Run tests
python ml-service/test_predictions.py
```

### Step 3: Manual Testing

#### Test Population Prediction

```bash
# Via backend
curl "http://localhost:4000/api/predictions/population?country=USA&years_ahead=5&base_year=2020"

# Direct ML service
curl "http://localhost:5000/api/predict/population?country=USA&years_ahead=5&base_year=2020"
```

#### Test Migration Prediction

```bash
# Via backend
curl -X POST "http://localhost:4000/api/predictions/migration" \
  -H "Content-Type: application/json" \
  -d '{
    "countries": ["USA", "MEX", "CAN"],
    "target_year": 2025,
    "base_year": 2020
  }'

# Direct ML service
curl -X POST "http://localhost:5000/api/predict/migration" \
  -H "Content-Type: application/json" \
  -d '{
    "countries": ["USA", "MEX", "CAN"],
    "target_year": 2025,
    "base_year": 2020
  }'
```

#### Test Indicator Prediction (All Indicators)

```bash
# Predict GDP
curl "http://localhost:4000/api/predictions/indicator?indicator=NY.GDP.MKTP.CD&countries=USA,MEX,CAN&target_year=2025&base_year=2020"

# Predict Life Expectancy
curl "http://localhost:4000/api/predictions/indicator?indicator=SP.DYN.LE00.IN&countries=USA,MEX,CAN&target_year=2025&base_year=2020"

# Predict CO2 Emissions
curl "http://localhost:4000/api/predictions/indicator?indicator=CC.CO2.EMSE.EL&countries=USA,CHN,IND&target_year=2025&base_year=2020"

# Via POST
curl -X POST "http://localhost:4000/api/predictions/indicator" \
  -H "Content-Type: application/json" \
  -d '{
    "indicator": "NY.GDP.MKTP.CD",
    "countries": ["USA", "MEX", "CAN"],
    "target_year": 2025,
    "base_year": 2020
  }'
```

## Using Predictions in Frontend

### Population Prediction

```typescript
import { predictPopulation } from "./services/api";

const result = await predictPopulation({
  country: "USA",
  years_ahead: 5,
  base_year: 2020,
});

// result.predictions = { 2021: 332000000, 2022: 335000000, ... }
```

### Migration Prediction

```typescript
import { predictMigration } from "./services/api";

const result = await predictMigration({
  countries: ["USA", "MEX", "CAN"],
  target_year: 2025,
  base_year: 2020,
});

// result.predictions = { "USA_MEX": 150000, "MEX_USA": 200000, ... }
```

### Indicator Prediction (New)

Add to `frontend/src/services/api.ts`:

```typescript
export async function predictIndicator({
  indicator,
  countries,
  target_year,
  base_year,
}: {
  indicator: string;
  countries: string[];
  target_year?: number;
  base_year?: number;
}): Promise<{
  indicator: string;
  countries: string[];
  target_year?: number;
  base_year?: number;
  predictions: Record<string, number>;
}> {
  const res = await fetch(`${API_BASE}/api/predictions/indicator`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      indicator,
      countries,
      target_year,
      base_year,
    }),
  });
  
  if (!res.ok) {
    throw new Error(`Indicator prediction request failed: ${res.status}`);
  }
  return res.json();
}
```

## Supported Indicators

All World Bank indicators are supported! Examples:

- **Economic**: `NY.GDP.MKTP.CD`, `NY.GDP.PCAP.CD`, `NY.GDP.MKTP.KD.ZG`
- **Population**: `SP.POP.TOTL`, `SP.URB.TOTL.IN.ZS`
- **Health**: `SP.DYN.LE00.IN`, `SH.XPD.CHEX.GD.ZS`
- **Education**: `SE.XPD.TOTL.GD.ZS`, `SE.PRM.ENRR`
- **Environment**: `CC.CO2.EMSE.EL`, `AG.LND.FRST.ZS`
- **Technology**: `IT.NET.USER.ZS`, `EG.USE.ELEC.KH.PC`
- **Trade**: `NE.TRD.GNFS.ZS`, `BX.KLT.DINV.WD.GD.ZS`

## Troubleshooting

### Training Issues

1. **No training data**: Check that countries and years have data available
2. **Memory errors**: Reduce number of countries or use smaller batch size
3. **Slow training**: Use GPU (`--device cuda`) or reduce epochs

### Prediction Issues

1. **Service not responding**: Check `docker-compose ps` and logs
2. **Empty predictions**: Model may be untrained, run training first
3. **Cache issues**: Clear Redis cache or disable caching

### Model Performance

- **Low accuracy**: Train with more countries and years
- **Overfitting**: Reduce model complexity or add dropout
- **Underfitting**: Increase model capacity or training epochs

## Next Steps

1. **Collect more data**: Integrate actual migration flow databases
2. **Improve features**: Add geographic distance, language similarity, etc.
3. **Tune hyperparameters**: Learning rate, model architecture, etc.
4. **Add validation**: Cross-validation, time-series validation
5. **Deploy to production**: Use trained model in production environment

