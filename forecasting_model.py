"""
Demand Forecasting Module
Implements multiple time-series forecasting models for demand prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Machine learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

from config import (
    FORECAST_HORIZON_DAYS, MIN_HISTORICAL_DAYS, 
    ARIMA_ORDER, XGBOOST_PARAMS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    Main class for demand forecasting using multiple models
    """
    
    def __init__(self, time_series_data: pd.DataFrame, item_code: str, item_description: str):
        """
        Initialize forecaster with time series data
        
        Args:
            time_series_data: DataFrame with 'date' and 'quantity' columns
            item_code: Item code for identification
            item_description: Item description
        """
        self.data = time_series_data.copy()
        self.item_code = item_code
        self.item_description = item_description
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        
    def validate_data(self) -> bool:
        """
        Validate time series data for forecasting
        
        Returns:
            Boolean indicating if data is valid
        """
        if self.data.empty:
            logger.error("Time series data is empty")
            return False
        
        if len(self.data) < MIN_HISTORICAL_DAYS:
            logger.warning(f"Insufficient data: {len(self.data)} days (minimum: {MIN_HISTORICAL_DAYS})")
            return False
        
        # Check for required columns
        if 'date' not in self.data.columns or 'quantity' not in self.data.columns:
            logger.error("Required columns 'date' and 'quantity' not found")
            return False
        
        return True
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for modeling
        
        Returns:
            Prepared DataFrame
        """
        df = self.data.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Remove any negative quantities
        df['quantity'] = df['quantity'].clip(lower=0)
        
        # Set date as index
        df = df.set_index('date')
        
        return df
    
    def train_arima_model(self, order: Tuple[int, int, int] = ARIMA_ORDER) -> Dict:
        """
        Train ARIMA model for forecasting
        
        Args:
            order: ARIMA (p, d, q) parameters
            
        Returns:
            Dictionary with model and forecast results
        """
        logger.info(f"Training ARIMA{order} model...")
        
        try:
            df = self.prepare_data()
            
            # Train ARIMA model
            model = ARIMA(df['quantity'], order=order)
            fitted_model = model.fit()
            
            # Make forecast
            forecast = fitted_model.forecast(steps=FORECAST_HORIZON_DAYS)
            forecast = np.maximum(forecast, 0)  # Ensure non-negative
            
            # Calculate in-sample metrics
            predictions = fitted_model.fittedvalues
            actual = df['quantity']
            
            # Align predictions and actual values
            common_index = predictions.index.intersection(actual.index)
            predictions = predictions.loc[common_index]
            actual = actual.loc[common_index]
            
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            
            # Create forecast DataFrame
            last_date = df.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=FORECAST_HORIZON_DAYS,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecasted_quantity': forecast,
                'model': 'ARIMA'
            })
            
            result = {
                'model': fitted_model,
                'forecast': forecast_df,
                'rmse': rmse,
                'mae': mae,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            self.models['arima'] = fitted_model
            self.forecasts['arima'] = forecast_df
            self.metrics['arima'] = {'rmse': rmse, 'mae': mae}
            
            logger.info(f"ARIMA model trained. RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            return {'error': str(e)}
    
    def train_prophet_model(self) -> Dict:
        """
        Train Prophet model for forecasting
        
        Returns:
            Dictionary with model and forecast results
        """
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available")
            return {'error': 'Prophet not installed'}
        
        logger.info("Training Prophet model...")
        
        try:
            df = self.prepare_data().reset_index()
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_df = df[['date', 'quantity']].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_df)
            
            # Make forecast
            future = model.make_future_dataframe(periods=FORECAST_HORIZON_DAYS)
            forecast = model.predict(future)
            
            # Extract future predictions
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(FORECAST_HORIZON_DAYS)
            forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)
            forecast_df = forecast_df.rename(columns={
                'ds': 'date',
                'yhat': 'forecasted_quantity',
                'yhat_lower': 'lower_bound',
                'yhat_upper': 'upper_bound'
            })
            forecast_df['model'] = 'Prophet'
            
            # Calculate metrics on historical data
            historical_forecast = forecast[['ds', 'yhat']].iloc[:-FORECAST_HORIZON_DAYS]
            historical_forecast = historical_forecast.merge(prophet_df, on='ds')
            
            rmse = np.sqrt(mean_squared_error(historical_forecast['y'], historical_forecast['yhat']))
            mae = mean_absolute_error(historical_forecast['y'], historical_forecast['yhat'])
            
            result = {
                'model': model,
                'forecast': forecast_df,
                'rmse': rmse,
                'mae': mae
            }
            
            self.models['prophet'] = model
            self.forecasts['prophet'] = forecast_df
            self.metrics['prophet'] = {'rmse': rmse, 'mae': mae}
            
            logger.info(f"Prophet model trained. RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            return {'error': str(e)}
    
    def train_ml_model(self, model_type: str = 'gradient_boosting') -> Dict:
        """
        Train machine learning model for forecasting
        
        Args:
            model_type: Type of ML model ('random_forest' or 'gradient_boosting')
            
        Returns:
            Dictionary with model and forecast results
        """
        logger.info(f"Training {model_type} model...")
        
        try:
            df = self.prepare_data().reset_index()
            
            # Create lag features
            for lag in [1, 7, 14, 30]:
                df[f'lag_{lag}'] = df['quantity'].shift(lag)
            
            # Create rolling statistics
            df['rolling_mean_7'] = df['quantity'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7'] = df['quantity'].rolling(window=7, min_periods=1).std()
            df['rolling_mean_30'] = df['quantity'].rolling(window=30, min_periods=1).mean()
            
            # Create date features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            
            # Drop rows with NaN (from lag features)
            df = df.dropna()
            
            if len(df) < 30:
                return {'error': 'Insufficient data after feature engineering'}
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col not in ['date', 'quantity']]
            X = df[feature_cols]
            y = df['quantity']
            
            # Split data
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # gradient_boosting
                model = GradientBoostingRegressor(**XGBOOST_PARAMS)
            
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Make future predictions
            last_row = df.iloc[-1:].copy()
            forecast_list = []
            
            for i in range(FORECAST_HORIZON_DAYS):
                # Prepare features for next day
                next_features = last_row[feature_cols].copy()
                
                # Predict
                next_pred = model.predict(next_features)[0]
                next_pred = max(0, next_pred)  # Ensure non-negative
                
                # Store prediction
                next_date = last_row['date'].values[0] + timedelta(days=1)
                forecast_list.append({
                    'date': next_date,
                    'forecasted_quantity': next_pred
                })
                
                # Update features for next iteration
                # This is a simplified approach; in practice, you'd update all lag features
                last_row['date'] = next_date
                last_row['quantity'] = next_pred
                
                # Update lag features
                for lag in [1, 7, 14, 30]:
                    if i >= lag:
                        last_row[f'lag_{lag}'] = forecast_list[i - lag]['forecasted_quantity']
            
            forecast_df = pd.DataFrame(forecast_list)
            forecast_df['model'] = model_type
            
            result = {
                'model': model,
                'forecast': forecast_df,
                'rmse': rmse,
                'mae': mae,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            self.models[model_type] = model
            self.forecasts[model_type] = forecast_df
            self.metrics[model_type] = {'rmse': rmse, 'mae': mae}
            
            logger.info(f"{model_type} model trained. RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
            return {'error': str(e)}
    
    def get_ensemble_forecast(self) -> pd.DataFrame:
        """
        Create ensemble forecast by averaging multiple models
        
        Returns:
            DataFrame with ensemble forecast
        """
        if not self.forecasts:
            logger.warning("No forecasts available for ensemble")
            return pd.DataFrame()
        
        logger.info("Creating ensemble forecast...")
        
        # Combine all forecasts
        all_forecasts = []
        for model_name, forecast_df in self.forecasts.items():
            df = forecast_df[['date', 'forecasted_quantity']].copy()
            df['model'] = model_name
            all_forecasts.append(df)
        
        combined = pd.concat(all_forecasts, ignore_index=True)
        
        # Calculate ensemble (average)
        ensemble = combined.groupby('date')['forecasted_quantity'].mean().reset_index()
        ensemble['model'] = 'Ensemble'
        ensemble['forecasted_quantity'] = ensemble['forecasted_quantity'].clip(lower=0)
        
        self.forecasts['ensemble'] = ensemble
        
        return ensemble
    
    def forecast(self, models: List[str] = ['arima', 'prophet', 'gradient_boosting']) -> Dict:
        """
        Run forecasting with specified models
        
        Args:
            models: List of model names to use
            
        Returns:
            Dictionary with all results
        """
        if not self.validate_data():
            return {'error': 'Data validation failed'}
        
        results = {}
        
        # Train specified models
        if 'arima' in models:
            results['arima'] = self.train_arima_model()
        
        if 'prophet' in models and PROPHET_AVAILABLE:
            results['prophet'] = self.train_prophet_model()
        
        if 'gradient_boosting' in models:
            results['gradient_boosting'] = self.train_ml_model('gradient_boosting')
        
        if 'random_forest' in models:
            results['random_forest'] = self.train_ml_model('random_forest')
        
        # Create ensemble if multiple models succeeded
        successful_forecasts = [k for k, v in results.items() if 'error' not in v]
        if len(successful_forecasts) > 1:
            ensemble = self.get_ensemble_forecast()
            results['ensemble'] = {'forecast': ensemble}
        
        return results


def forecast_demand(time_series_data: pd.DataFrame, item_code: str, 
                   item_description: str, models: List[str] = None) -> Dict:
    """
    Main function to forecast demand for an item
    
    Args:
        time_series_data: DataFrame with 'date' and 'quantity' columns
        item_code: Item code
        item_description: Item description
        models: List of models to use (default: ['arima', 'prophet', 'gradient_boosting'])
        
    Returns:
        Dictionary with forecast results
    """
    if models is None:
        models = ['arima', 'gradient_boosting']
        if PROPHET_AVAILABLE:
            models.append('prophet')
    
    forecaster = DemandForecaster(time_series_data, item_code, item_description)
    results = forecaster.forecast(models)
    
    return results
