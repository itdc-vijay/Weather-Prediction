import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta, datetime
from app.utils.preprocess import load_data, prepare_data_for_prediction, TARGET_FEATURES, TIMESTAMP_COL, LAG_FEATURES
# Import ProphetRegressor to ensure it's available when loading models
from app.ml.models import ProphetRegressor

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_model(city_name, model_name):
    model_filename = os.path.join(MODELS_DIR, f"{city_name}_{model_name}.pkl")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    model = joblib.load(model_filename)
    return model

def is_prophet_model(model):
    """Check if the model is a Prophet model"""
    return hasattr(model, 'models') and hasattr(model, 'predict') and not hasattr(model, 'estimators_')

def calculate_extended_periods(extended_option):
    """Calculate number of hours based on extended forecast option"""
    if extended_option == "1month":
        return 24 * 30  # ~30 days
    elif extended_option == "3months":
        return 24 * 90  # ~90 days
    elif extended_option == "6months":
        return 24 * 180  # ~180 days
    elif extended_option == "1year":
        return 24 * 365  # ~365 days
    return None  # No extension

def make_predictions_with_prophet(model, city_name, hours_to_predict, include_bounds=False):
    """Generate predictions using Prophet model with extended capabilities"""
    print(f"Using Prophet-specific prediction path for {city_name}")
    
    # Get the last timestamp from historical data
    df_history = load_data(city_name)
    if df_history.empty:
        raise ValueError(f"No historical data found for {city_name} to make predictions.")
    
    last_timestamp = df_history.index.max()
    
    # Create future dates DataFrame
    future_dates = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=hours_to_predict,
        freq='H'
    )
    
    # Prepare predictions dataframe
    predictions = []
    
    # For each target feature (temperature, humidity, etc.)
    for i, feature in enumerate(TARGET_FEATURES):
        # Check if we have a model for this feature
        if i < len(model.models):
            # Create future dataframe for Prophet
            future = pd.DataFrame({'ds': future_dates})
            # Make prediction
            forecast = model.models[i].predict(future)
            
            # Extract the predicted values
            for j, timestamp in enumerate(future_dates):
                if j >= len(predictions):
                    predictions.append({TIMESTAMP_COL: timestamp})
                
                # Add the main forecast
                predictions[j][feature] = forecast.iloc[j]['yhat']
                
                # Add uncertainty bounds if requested
                if include_bounds:
                    predictions[j][f"{feature}_lower"] = forecast.iloc[j]['yhat_lower']
                    predictions[j][f"{feature}_upper"] = forecast.iloc[j]['yhat_upper']
    
    # Apply constraints to the predictions
    for pred in predictions:
        if "Humidity (%)" in pred:
            pred["Humidity (%)"] = np.clip(pred["Humidity (%)"], 0, 100)
            if include_bounds and "Humidity (%)_lower" in pred:
                pred["Humidity (%)_lower"] = np.clip(pred["Humidity (%)_lower"], 0, 100)
                pred["Humidity (%)_upper"] = np.clip(pred["Humidity (%)_upper"], 0, 100)
                
        if "Wind Direction (°)" in pred:
            pred["Wind Direction (°)"] = np.clip(pred["Wind Direction (°)"], 0, 360)
            if include_bounds and "Wind Direction (°)_lower" in pred:
                pred["Wind Direction (°)_lower"] = np.clip(pred["Wind Direction (°)_lower"], 0, 360)
                pred["Wind Direction (°)_upper"] = np.clip(pred["Wind Direction (°)_upper"], 0, 360)
    
    print(f"Finished Prophet prediction. Generated {len(predictions)} data points.")
    return pd.DataFrame(predictions)

def make_predictions(city_name, model_name, hours_to_predict=48, include_bounds=False, prophet_extended=None):
    """Enhanced prediction function with Prophet-specific options"""
    model = load_model(city_name, model_name)
    
    # Check if this is a Prophet model with extended forecast
    if (model_name == "Prophet" or is_prophet_model(model)) and prophet_extended:
        extended_hours = calculate_extended_periods(prophet_extended)
        if extended_hours:
            hours_to_predict = extended_hours
            print(f"Using extended Prophet forecast: {prophet_extended} ({hours_to_predict} hours)")
    
    # Prophet-specific prediction path
    if model_name == "Prophet" or is_prophet_model(model):
        return make_predictions_with_prophet(model, city_name, hours_to_predict, include_bounds)
    
    # Standard prediction for tree-based models (existing code)
    df_history = load_data(city_name)

    if df_history.empty:
        raise ValueError(f"No historical data found for {city_name} to make predictions.")

    predictions = []
    current_history = df_history.copy()
    last_timestamp = current_history.index.max()

    print(f"Starting prediction for {city_name} using {model_name} for {hours_to_predict} hours.")
    print(f"Latest data point timestamp: {last_timestamp}")

    for i in range(hours_to_predict):
        try:
            X_pred_input = prepare_data_for_prediction(current_history, lag_features=LAG_FEATURES)
        except ValueError as e:
             print(f"Error preparing data at step {i+1}/{hours_to_predict}: {e}")
             print(f"History length: {len(current_history)}")
             break

        next_hour_pred_values = model.predict(X_pred_input)[0]

        next_timestamp = last_timestamp + timedelta(hours=i + 1)
        pred_record = {TIMESTAMP_COL: next_timestamp}
        for idx, feature in enumerate(TARGET_FEATURES):
             if feature == "Humidity (%)":
                 pred_record[feature] = np.clip(next_hour_pred_values[idx], 0, 100)
             elif feature == "Wind Direction (°)":
                 pred_record[feature] = np.clip(next_hour_pred_values[idx], 0, 360)
             else:
                 pred_record[feature] = next_hour_pred_values[idx]
        predictions.append(pred_record)

        new_row_df = pd.DataFrame([pred_record])
        new_row_df[TIMESTAMP_COL] = pd.to_datetime(new_row_df[TIMESTAMP_COL])
        new_row_df.set_index(TIMESTAMP_COL, inplace=True)
        new_row_df = new_row_df[TARGET_FEATURES]

        current_history = pd.concat([current_history, new_row_df])

    print(f"Finished prediction. Generated {len(predictions)} data points.")
    return pd.DataFrame(predictions)
