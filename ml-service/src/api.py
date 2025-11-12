"""
FastAPI service for ML predictions.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import redis
import os
import logging

from .predictor import MigrationPredictor
from .indicator_predictor import IndicatorPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="World Bank ML Prediction Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis cache
redis_client = None
try:
    redis_host = os.getenv("REDIS_HOST", "cache")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    
    redis_kwargs = {
        "host": redis_host,
        "port": redis_port,
        "decode_responses": True,
        "socket_connect_timeout": 5,
    }
    
    if redis_password:
        redis_kwargs["password"] = redis_password
    
    redis_client = redis.Redis(**redis_kwargs)
    redis_client.ping()
    logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}. Continuing without cache.")

# Model path
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/migration_gnn.pth")

# Initialize predictors
predictor = MigrationPredictor(
    model_path=MODEL_PATH,
    device=os.getenv("DEVICE", "cpu"),
    cache=redis_client,
)

indicator_predictor = IndicatorPredictor(
    model_path=MODEL_PATH,
    device=os.getenv("DEVICE", "cpu"),
    cache=redis_client,
)

# Auto-train model on startup if model doesn't exist or AUTO_TRAIN is enabled
# Use FastAPI startup event to run training in background
AUTO_TRAIN = os.getenv("AUTO_TRAIN", "false").lower() == "true"

@app.on_event("startup")
async def startup_event():
    """Startup event handler - runs after API is ready."""
    if AUTO_TRAIN or not Path(MODEL_PATH).exists():
        logger.info("Scheduling automatic model training in background...")
        
        def train_in_background():
            """Train model in background thread."""
            import time
            
            # Wait a bit for API to fully start
            time.sleep(3)
            
            try:
                from .auto_train import auto_train
                logger.info("Starting background model training...")
                training_success = auto_train(
                    model_path=MODEL_PATH,
                    epochs=int(os.getenv("TRAINING_EPOCHS", "30")),
                    cache=redis_client,
                )
                if training_success:
                    logger.info("Model training completed. Reloading predictors...")
                    # Reload predictors with new model
                    global predictor, indicator_predictor
                    predictor = MigrationPredictor(
                        model_path=MODEL_PATH,
                        device=os.getenv("DEVICE", "cpu"),
                        cache=redis_client,
                    )
                    indicator_predictor = IndicatorPredictor(
                        model_path=MODEL_PATH,
                        device=os.getenv("DEVICE", "cpu"),
                        cache=redis_client,
                    )
                    logger.info("Predictors reloaded with trained model.")
                else:
                    logger.warning("Model training failed. Using untrained model.")
            except Exception as e:
                logger.error(f"Auto-training error: {e}", exc_info=True)
        
        # Start training in background thread
        import threading
        training_thread = threading.Thread(target=train_in_background, daemon=True)
        training_thread.start()
        logger.info("Background training thread started. API is ready to accept requests.")


class PredictionRequest(BaseModel):
    countries: List[str]
    target_year: Optional[int] = None
    base_year: Optional[int] = None


class PopulationPredictionRequest(BaseModel):
    country: str
    years_ahead: int = 5
    base_year: Optional[int] = None


class IndicatorPredictionRequest(BaseModel):
    indicator: str
    countries: List[str]
    target_year: Optional[int] = None
    base_year: Optional[int] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ml-prediction"}


@app.post("/api/predict/migration")
async def predict_migration(request: PredictionRequest):
    """
    Predict migration flows between countries.
    
    Args:
        request: Prediction request with countries and optional years
    
    Returns:
        Dictionary mapping (source, target) country pairs to predicted migration volumes
    """
    try:
        predictions = predictor.predict_migration_flows(
            countries=request.countries,
            target_year=request.target_year,
            base_year=request.base_year,
        )
        
        # Convert tuple keys to strings for JSON serialization
        result = {
            f"{src}_{tgt}": volume
            for (src, tgt), volume in predictions.items()
        }
        
        return {
            "countries": request.countries,
            "target_year": request.target_year,
            "predictions": result,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/population")
async def predict_population(request: PopulationPredictionRequest):
    """
    Predict population growth for a country.
    
    Args:
        request: Population prediction request
    
    Returns:
        Dictionary mapping years to predicted population values
    """
    try:
        predictions = predictor.predict_population_growth(
            country=request.country,
            years_ahead=request.years_ahead,
            base_year=request.base_year,
        )
        
        return {
            "country": request.country,
            "base_year": request.base_year,
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(f"Population prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/population")
async def predict_population_get(
    country: str = Query(..., description="Country ISO3 code"),
    years_ahead: int = Query(5, description="Number of years to predict ahead"),
    base_year: Optional[int] = Query(None, description="Base year for prediction"),
):
    """
    Predict population growth for a country (GET endpoint).
    """
    try:
        predictions = predictor.predict_population_growth(
            country=country.upper(),
            years_ahead=years_ahead,
            base_year=base_year,
        )
        
        return {
            "country": country.upper(),
            "base_year": base_year,
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(f"Population prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/indicator")
async def predict_indicator(request: IndicatorPredictionRequest):
    """
    Predict any World Bank indicator for multiple countries.
    
    Body: {
        "indicator": "NY.GDP.MKTP.CD",
        "countries": ["USA", "MEX", "CAN"],
        "target_year": 2025,
        "base_year": 2020
    }
    """
    try:
        predictions = indicator_predictor.predict_indicator(
            indicator=request.indicator.upper(),
            countries=[c.upper() for c in request.countries],
            target_year=request.target_year,
            base_year=request.base_year,
        )
        
        return {
            "indicator": request.indicator.upper(),
            "countries": [c.upper() for c in request.countries],
            "target_year": request.target_year,
            "base_year": request.base_year,
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(f"Indicator prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/indicator")
async def predict_indicator_get(
    indicator: str = Query(..., description="World Bank indicator code"),
    countries: str = Query(..., description="Comma-separated country codes"),
    target_year: Optional[int] = Query(None, description="Target year"),
    base_year: Optional[int] = Query(None, description="Base year"),
):
    """
    Predict any World Bank indicator (GET endpoint).
    """
    try:
        country_list = [c.strip().upper() for c in countries.split(",")]
        
        predictions = indicator_predictor.predict_indicator(
            indicator=indicator.upper(),
            countries=country_list,
            target_year=target_year,
            base_year=base_year,
        )
        
        return {
            "indicator": indicator.upper(),
            "countries": country_list,
            "target_year": target_year,
            "base_year": base_year,
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(f"Indicator prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class TrainingRequest(BaseModel):
    countries: Optional[List[str]] = None
    years: Optional[List[int]] = None
    epochs: int = 30


@app.post("/api/train")
async def train_model(request: Optional[TrainingRequest] = None):
    """
    Manually trigger model training.
    
    Body (optional): {
        "countries": ["USA", "MEX", ...],
        "years": [2000, 2001, ...],
        "epochs": 50
    }
    """
    try:
        from .auto_train import auto_train
        
        countries = request.countries if request else None
        years = request.years if request else None
        epochs = request.epochs if request else 30
        
        logger.info("Manual training triggered")
        
        success = auto_train(
            model_path=MODEL_PATH,
            countries=countries,
            years=years,
            epochs=epochs,
            cache=redis_client,
        )
        
        if success:
            # Reload predictors
            global predictor, indicator_predictor
            predictor = MigrationPredictor(
                model_path=MODEL_PATH,
                device=os.getenv("DEVICE", "cpu"),
                cache=redis_client,
            )
            indicator_predictor = IndicatorPredictor(
                model_path=MODEL_PATH,
                device=os.getenv("DEVICE", "cpu"),
                cache=redis_client,
            )
            
            return {
                "status": "success",
                "message": "Model trained and reloaded",
                "model_path": MODEL_PATH,
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
            
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

