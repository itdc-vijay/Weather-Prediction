import pandas as pd
import numpy as np
from prophet import Prophet
import warnings

# Suppress Prophet related warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Prophet wrapper for multi-output regression compatibility
class ProphetRegressor:
    def __init__(self, seasonality_mode='multiplicative', yearly_seasonality=True, 
                 weekly_seasonality=True, daily_seasonality=True):
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.models = None
        
    def fit(self, X, y):
        # Create a separate Prophet model for each target feature
        self.models = []
        for i in range(y.shape[1]):
            model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            df = pd.DataFrame({
                'ds': X.index,  # dates
                'y': y.iloc[:, i]  # target variable
            })
            model.fit(df)
            self.models.append(model)
        return self
    
    def predict(self, X):
        if self.models is None:
            raise ValueError("Model has not been fitted yet.")
        
        predictions = []
        for model in self.models:
            # Create future dataframe with 'ds' column
            future = pd.DataFrame({'ds': X.index})
            forecast = model.predict(future)
            predictions.append(forecast['yhat'].values)
        
        # Combine predictions for all features
        return np.column_stack(predictions)
