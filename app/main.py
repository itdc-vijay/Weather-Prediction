from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from typing import Optional

from app.ml.predict import make_predictions
from app.ml.ensemble import predict_ensemble, BASE_MODEL_NAMES
from app.utils.preprocess import TARGET_FEATURES, TIMESTAMP_COL
from app.ml.evaluate import load_model_metrics, get_all_metrics, evaluate_ensemble

app = FastAPI(title="Weather Forecast API")

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def filter_by_day(df: pd.DataFrame, day_of_week: int) -> pd.DataFrame:
    if TIMESTAMP_COL not in df.columns:
         df[TIMESTAMP_COL] = pd.to_datetime(df.index)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df[df[TIMESTAMP_COL].dt.dayofweek == day_of_week].copy()

@app.get("/predict")
async def get_prediction(
    city: str = Query(..., description="City name (e.g., ahmedabad)"),
    model_name: str = Query(..., description=f"Model name (e.g., {' / '.join(BASE_MODEL_NAMES)} / Ensemble)"),
    forecast_type: str = Query(..., description="Forecast duration ('48h', '1week' or '2weeks')"),
    day_of_week: Optional[int] = Query(None, description="Day of week (0=Mon, 6=Sun) - only if forecast_type is '1week' or '2weeks'"),
    # Prophet-specific parameters
    prophet_extended: Optional[str] = Query(None, description="Extended forecast period for Prophet (1month, 3months, 6months, 1year)"),
    include_bounds: Optional[bool] = Query(False, description="Include Prophet's uncertainty bounds")
):
    allowed_cities = ["ahmedabad", "mumbai", "delhi", "bengaluru"]
    allowed_models = BASE_MODEL_NAMES + ["Ensemble"]
    allowed_forecast_types = ["48h", "1week", "2weeks"]
    allowed_prophet_extended = ["1month", "3months", "6months", "1year", None]

    # Validation checks
    if city not in allowed_cities:
        raise HTTPException(status_code=400, detail=f"Invalid city. Allowed: {', '.join(allowed_cities)}")
    if model_name not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Allowed: {', '.join(allowed_models)}")
    if forecast_type not in allowed_forecast_types:
        raise HTTPException(status_code=400, detail=f"Invalid forecast type. Allowed: {', '.join(allowed_forecast_types)}")
    if forecast_type == "48h" and day_of_week is not None:
        raise HTTPException(status_code=400, detail="Day of week selection is only valid for '1week' or '2weeks' forecast type.")
    if day_of_week is not None and not (0 <= day_of_week <= 6):
        raise HTTPException(status_code=400, detail="Invalid day_of_week. Must be between 0 (Monday) and 6 (Sunday).")
    if prophet_extended and prophet_extended not in allowed_prophet_extended:
        raise HTTPException(status_code=400, detail=f"Invalid prophet_extended. Allowed: {', '.join([p for p in allowed_prophet_extended if p])}")
    if prophet_extended and model_name != "Prophet":
        raise HTTPException(status_code=400, detail="Extended forecasting is only available with the Prophet model.")
    if include_bounds and model_name != "Prophet":
        raise HTTPException(status_code=400, detail="Uncertainty bounds are only available with the Prophet model.")

    # Determine forecast length
    hours_to_predict = 48 if forecast_type == "48h" else 168 if forecast_type == "1week" else 336

    try:
        print(f"Received request: city={city}, model={model_name}, type={forecast_type}, day={day_of_week}, prophet_extended={prophet_extended}, include_bounds={include_bounds}")
        
        if model_name == "Ensemble":
            # Ensemble doesn't support Prophet extensions
            df_predictions = predict_ensemble(city, hours_to_predict)
        else:
            # Pass Prophet-specific parameters when applicable
            df_predictions = make_predictions(
                city, 
                model_name, 
                hours_to_predict,
                include_bounds=include_bounds,
                prophet_extended=prophet_extended
            )

        if df_predictions.empty:
            raise HTTPException(status_code=500, detail="Prediction generation failed or returned empty results.")

        df_predictions[TIMESTAMP_COL] = df_predictions[TIMESTAMP_COL].dt.strftime('%Y-%m-%d %H:%M:%S')

        if forecast_type in ["1week", "2weeks"] and day_of_week is not None:
            df_predictions = filter_by_day(df_predictions, day_of_week)
            if df_predictions.empty:
                print(f"Warning: No predictions found for day_of_week={day_of_week} within the forecast period.")
                return []

        result = df_predictions.round(2).to_dict(orient='records')
        return result

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Weather Forecast API!"}

@app.get("/model-metrics")
async def get_model_metrics(
    city: Optional[str] = Query(None, description="City name (e.g., ahmedabad). If not provided, returns metrics for all cities."),
    model_name: Optional[str] = Query(None, description="Model name. If not provided, returns metrics for all models.")
):
    allowed_cities = ["ahmedabad", "mumbai", "delhi", "bengaluru"]
    allowed_models = BASE_MODEL_NAMES + ["Ensemble"]
    
    # Validate parameters if provided
    if city and city not in allowed_cities:
        raise HTTPException(status_code=400, detail=f"Invalid city. Allowed: {', '.join(allowed_cities)}")
    if model_name and model_name not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Allowed: {', '.join(allowed_models)}")
    
    try:
        # Special handling for Ensemble model
        if city and model_name == "Ensemble":
            metrics_file = os.path.join(os.path.dirname(__file__), 'metrics', f"{city}_Ensemble_metrics.json")
            
            # If metrics don't exist, generate them
            if not os.path.exists(metrics_file):
                print(f"Generating Ensemble metrics for {city}...")
                metrics = evaluate_ensemble(city)
                return metrics
            
            # Load existing metrics
            metrics = load_model_metrics(city, "Ensemble")
            if not metrics:
                print(f"Regenerating Ensemble metrics for {city}...")
                metrics = evaluate_ensemble(city)
            return metrics
            
        # If both city and model are specified, return specific metrics
        if city and model_name:
            metrics = load_model_metrics(city, model_name)
            if not metrics:
                raise HTTPException(status_code=404, detail=f"No metrics found for {model_name} in {city}")
            return metrics
        
        # Otherwise return all metrics or filtered metrics
        all_metrics = get_all_metrics()
        
        # Filter by city if provided
        if city:
            if city not in all_metrics:
                return {}
            return {city: all_metrics[city]}
        
        # Filter by model if provided
        if model_name:
            filtered_metrics = {}
            for city_name, models in all_metrics.items():
                if model_name in models:
                    if city_name not in filtered_metrics:
                        filtered_metrics[city_name] = {}
                    filtered_metrics[city_name][model_name] = models[model_name]
            return filtered_metrics
        
        # Return all metrics if no filters
        return all_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model metrics: {str(e)}")