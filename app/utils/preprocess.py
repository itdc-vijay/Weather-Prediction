import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TARGET_FEATURES = ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Wind Direction (°)"]
TIMESTAMP_COL = "Timestamp"
LAG_FEATURES = 24

def load_data(city_name):
    file_path = os.path.join(DATA_DIR, f"{city_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found for city: {city_name} at {file_path}")
    df = pd.read_csv(file_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df.sort_values(TIMESTAMP_COL, inplace=True)
    df.set_index(TIMESTAMP_COL, inplace=True)
    for col in TARGET_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {city_name}.csv")
    return df[TARGET_FEATURES]

def create_features(df, lag_features=LAG_FEATURES):
    df_feat = df.copy()
    for col in TARGET_FEATURES:
        for lag in range(1, lag_features + 1):
            df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)

    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear

    df_feat.dropna(inplace=True)
    return df_feat

def prepare_data_for_training(city_name, lag_features=LAG_FEATURES):
    df = load_data(city_name)
    df_processed = create_features(df, lag_features)

    X = df_processed.drop(columns=TARGET_FEATURES)
    y = df_processed[TARGET_FEATURES]

    return X, y

def prepare_data_for_prediction(df_history, lag_features=LAG_FEATURES):
    if len(df_history) < lag_features:
         raise ValueError(f"Need at least {lag_features} hours of history for prediction.")
    recent_data = df_history.iloc[-lag_features:].copy()
    df_feat = create_features(recent_data, lag_features)
    if df_feat.empty:
         temp_data = df_history.iloc[-(lag_features+1):].copy()
         df_feat_temp = create_features(temp_data, lag_features)
         if df_feat_temp.empty:
              raise ValueError("Could not create valid feature row for prediction.")
         return df_feat_temp.iloc[-1:].drop(columns=TARGET_FEATURES)

    return df_feat.iloc[-1:].drop(columns=TARGET_FEATURES)