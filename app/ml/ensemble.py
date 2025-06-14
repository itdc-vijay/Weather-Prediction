import pandas as pd
import numpy as np
import os
from app.ml.predict import make_predictions, TARGET_FEATURES, TIMESTAMP_COL # Reuse prediction logic

# List of base models used for the ensemble
BASE_MODEL_NAMES = ["LightGBM", "CatBoost", "ExtraTrees", "XGBoost", "HistGradientBoosting", "Prophet"]

def predict_ensemble(city_name, hours_to_predict=48):
    """Generates predictions from all base models and averages them."""
    all_predictions = []
    print(f"Starting ensemble prediction for {city_name} for {hours_to_predict} hours.")

    for model_name in BASE_MODEL_NAMES:
        try:
            print(f"Generating predictions using: {model_name}")
            df_pred = make_predictions(city_name, model_name, hours_to_predict)
            if not df_pred.empty:
                df_pred.set_index(TIMESTAMP_COL, inplace=True)
                all_predictions.append(df_pred)
            else:
                print(f"⚠️ Received empty predictions from {model_name} for {city_name}. Excluding from ensemble.")
        except FileNotFoundError:
            print(f"⚠️ Model file for {model_name} in {city_name} not found. Skipping.")
        except Exception as e:
            print(f"❌ Error getting predictions from {model_name} for {city_name}: {e}")

    if not all_predictions:
        raise ValueError(f"No base model predictions could be generated for ensemble in {city_name}.")

    # Concatenate predictions along a new axis (axis=0 stacks rows, axis=1 stacks columns)
    # We want to average across models for the same timestamp and feature.
    # Use pd.concat and then groupby index (timestamp) and mean.
    concat_df = pd.concat(all_predictions)
    ensemble_df = concat_df.groupby(concat_df.index).mean()

    # Reset index to have Timestamp as a column again
    ensemble_df.reset_index(inplace=True)

    # Ensure column order
    ensemble_df = ensemble_df[[TIMESTAMP_COL] + TARGET_FEATURES]

    print(f"Finished ensemble prediction. Averaged {len(all_predictions)} models.")
    return ensemble_df