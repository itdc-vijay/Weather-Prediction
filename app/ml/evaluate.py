import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from app.utils.preprocess import TARGET_FEATURES, prepare_data_for_training
from app.ml.ensemble import BASE_MODEL_NAMES 
import joblib

METRICS_DIR = os.path.join(os.path.dirname(__file__), '..', 'metrics')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(METRICS_DIR, exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Calculate various error metrics between true and predicted values."""
    metrics = {}
    
    # Calculate metrics for each target feature
    for i, feature in enumerate(TARGET_FEATURES):
        feature_metrics = {
            'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'r2': r2_score(y_true[:, i], y_pred[:, i])
        }
        
        # Calculate MAPE (Mean Absolute Percentage Error) with handling for zeros
        y_true_feature = y_true[:, i]
        y_pred_feature = y_pred[:, i]
        
        # Avoid division by zero in MAPE calculation
        mask = y_true_feature != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true_feature[mask] - y_pred_feature[mask]) / y_true_feature[mask])) * 100
            feature_metrics['mape'] = mape
        else:
            feature_metrics['mape'] = None
            
        metrics[feature] = feature_metrics
    
    # Calculate overall average metrics
    overall_mae = np.mean([metrics[feature]['mae'] for feature in TARGET_FEATURES])
    overall_rmse = np.mean([metrics[feature]['rmse'] for feature in TARGET_FEATURES])
    overall_r2 = np.mean([metrics[feature]['r2'] for feature in TARGET_FEATURES])
    
    mape_values = [metrics[feature]['mape'] for feature in TARGET_FEATURES if metrics[feature]['mape'] is not None]
    overall_mape = np.mean(mape_values) if mape_values else None
    
    metrics['overall'] = {
        'mae': overall_mae,
        'rmse': overall_rmse,
        'r2': overall_r2,
        'mape': overall_mape
    }
    
    return metrics

def evaluate_model(model, X, y, test_size=0.2, random_state=42):
    """Evaluate model performance using train-test split."""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert to numpy arrays if they aren't already
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    y_pred_np = np.array(y_pred)
    
    # Calculate metrics
    return calculate_metrics(y_test_np, y_pred_np)

def save_model_metrics(city, model_name, metrics):
    """Save model evaluation metrics to a JSON file."""
    metrics_filename = os.path.join(METRICS_DIR, f"{city}_{model_name}_metrics.json")
    
    # Convert numpy values to Python native types for JSON serialization
    serializable_metrics = {}
    for feature, feature_metrics in metrics.items():
        serializable_metrics[feature] = {
            metric_name: float(metric_value) if metric_value is not None and not np.isnan(metric_value) else None
            for metric_name, metric_value in feature_metrics.items()
        }
    
    with open(metrics_filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    return metrics_filename

def load_model_metrics(city, model_name):
    """Load model metrics from JSON file."""
    metrics_filename = os.path.join(METRICS_DIR, f"{city}_{model_name}_metrics.json")
    
    if not os.path.exists(metrics_filename):
        return None
    
    with open(metrics_filename, 'r') as f:
        return json.load(f)

def get_all_metrics():
    """Get metrics for all available city-model pairs."""
    all_metrics = {}
    
    if not os.path.exists(METRICS_DIR):
        return all_metrics
    
    for filename in os.listdir(METRICS_DIR):
        if filename.endswith('_metrics.json'):
            parts = filename.split('_')
            if len(parts) >= 3:
                city = parts[0]
                model_name = '_'.join(parts[1:-1])  # Handle model names with underscores
                
                if city not in all_metrics:
                    all_metrics[city] = {}
                    
                with open(os.path.join(METRICS_DIR, filename), 'r') as f:
                    all_metrics[city][model_name] = json.load(f)
    
    return all_metrics

def evaluate_ensemble(city, test_size=0.2, random_state=42):
    """Evaluate the ensemble method by averaging predictions from base models."""
    # Get training and test data
    X, y = prepare_data_for_training(city)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Convert test data to numpy arrays
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    
    # Store predictions from each model
    all_predictions = []
    
    # Get predictions from each base model
    for model_name in BASE_MODEL_NAMES:
        try:
            model_path = os.path.join(MODELS_DIR, f"{city}_{model_name}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
                all_predictions.append(y_pred)
                print(f"✅ Loaded predictions from {model_name} for ensemble evaluation")
            else:
                print(f"⚠️ Model {model_name} not found for {city}")
        except Exception as e:
            print(f"❌ Error getting predictions from {model_name}: {e}")
    
    if not all_predictions:
        raise ValueError(f"No base model predictions could be generated for ensemble evaluation in {city}")
    
    # Average predictions from all models
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_np, ensemble_pred)
    
    # Save metrics
    save_model_metrics(city, "Ensemble", metrics)
    
    return metrics
