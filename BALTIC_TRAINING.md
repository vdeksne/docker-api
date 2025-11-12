# Baltic Countries Training Configuration

The ML model has been configured to prioritize training on population data for Latvia (LVA), Lithuania (LTU), and Estonia (EST) to improve prediction accuracy.

## Changes Made

### 1. Training Data Prioritization
- **Baltic countries are listed first** in the training country list
- **Neighboring countries** (Poland, Sweden, Finland, Germany) are included for context
- **Extended year range**: 1990-2023 (34 years) instead of 2000-2020 (21 years)
- **Required Baltic presence**: Training samples require at least 2 Baltic countries to be present

### 2. Enhanced Historical Data Collection
- **For Baltic countries**: Uses 10 years of historical data (base_year - 10 to base_year)
- **For other countries**: Uses 5 years of historical data (standard)
- **Fallback mechanism**: If current year data is unavailable, uses previous year's data

### 3. Improved Prediction Logic
- **Single country predictions**: 
  - Baltic countries: Uses 10 years of historical trend data
  - Other countries: Uses 4 years of historical trend data
  - Weighted recent years: For Baltic countries, recent 5 years are weighted more heavily
  
- **Multiple country predictions**:
  - Baltic countries get more historical context
  - Better relationship modeling between Baltic countries

## Training Configuration

### Default Countries (Priority Order)
1. **LVA, EST, LTU** (Baltic countries - highest priority)
2. **POL, SWE, FIN, DEU** (Neighboring countries for context)
3. Other countries for broader training

### Year Range
- **1990-2023**: Extended range for more historical context
- Especially important for Baltic countries post-Soviet independence

## How It Works

1. **Data Collection**: 
   - Training prioritizes fetching Baltic country data first
   - If data for a year is missing, falls back to previous year
   - Requires at least 2 Baltic countries per training sample

2. **Model Training**:
   - Model learns patterns specific to Baltic countries
   - Better understanding of relationships between LVA, EST, LTU
   - More accurate predictions for these countries

3. **Prediction**:
   - When predicting for Baltic countries, uses more historical data
   - Better trend analysis with weighted recent years
   - More accurate growth rate calculations

## Monitoring Training

Check training progress:
```bash
docker-compose logs -f ml-service | grep -E "(Baltic|LVA|EST|LTU|Training|Prepared)"
```

## Manual Retraining

To force retraining with Baltic focus:
```bash
curl -X POST http://localhost:5001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "countries": ["LVA", "EST", "LTU", "POL", "SWE", "FIN"],
    "years": [1990, 1991, 1992, ..., 2023],
    "epochs": 50
  }'
```

## Expected Improvements

- **Better initial predictions** for Baltic countries
- **More accurate trend analysis** using extended historical data
- **Improved relationship modeling** between Baltic countries
- **Reduced prediction errors** at the beginning of time series

