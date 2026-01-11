"""
Unit tests for forecasting module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forecasting_model import (
    DemandForecaster,
    forecast_demand
)


@pytest.fixture
def sample_time_series():
    """Create sample time series data"""
    dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
    
    # Create seasonal pattern with trend
    trend = np.linspace(10, 15, 120)
    seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, 120))
    noise = np.random.normal(0, 1, 120)
    
    quantities = trend + seasonal + noise
    quantities = np.maximum(quantities, 0)  # Ensure non-negative
    
    return pd.DataFrame({
        'date': dates,
        'quantity': quantities
    })


def test_forecaster_initialization(sample_time_series):
    """Test forecaster initialization"""
    forecaster = DemandForecaster(
        sample_time_series,
        'TEST-001',
        'Test Item'
    )
    
    assert forecaster.item_code == 'TEST-001'
    assert forecaster.item_description == 'Test Item'
    assert len(forecaster.data) == 120


def test_data_validation_success(sample_time_series):
    """Test data validation with valid data"""
    forecaster = DemandForecaster(
        sample_time_series,
        'TEST-001',
        'Test Item'
    )
    
    is_valid = forecaster.validate_data()
    assert is_valid == True


def test_data_validation_insufficient_data():
    """Test validation with insufficient data"""
    small_data = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=30, freq='D'),
        'quantity': np.random.rand(30) * 10
    })
    
    forecaster = DemandForecaster(small_data, 'TEST-001', 'Test Item')
    is_valid = forecaster.validate_data()
    
    assert is_valid == False


def test_data_validation_empty():
    """Test validation with empty data"""
    empty_data = pd.DataFrame(columns=['date', 'quantity'])
    
    forecaster = DemandForecaster(empty_data, 'TEST-001', 'Test Item')
    is_valid = forecaster.validate_data()
    
    assert is_valid == False


def test_prepare_data(sample_time_series):
    """Test data preparation"""
    forecaster = DemandForecaster(
        sample_time_series,
        'TEST-001',
        'Test Item'
    )
    
    prepared = forecaster.prepare_data()
    
    assert isinstance(prepared.index, pd.DatetimeIndex)
    assert 'quantity' in prepared.columns
    assert prepared['quantity'].min() >= 0  # Non-negative


def test_train_arima_model(sample_time_series):
    """Test ARIMA model training"""
    forecaster = DemandForecaster(
        sample_time_series,
        'TEST-001',
        'Test Item'
    )
    
    result = forecaster.train_arima_model()
    
    assert 'model' in result
    assert 'forecast' in result
    assert 'rmse' in result
    assert 'mae' in result
    assert len(result['forecast']) == 30  # 30-day forecast


def test_train_ml_model(sample_time_series):
    """Test ML model training"""
    forecaster = DemandForecaster(
        sample_time_series,
        'TEST-001',
        'Test Item'
    )
    
    result = forecaster.train_ml_model('gradient_boosting')
    
    assert 'model' in result
    assert 'forecast' in result
    assert 'rmse' in result
    assert 'mae' in result


def test_forecast_function(sample_time_series):
    """Test main forecast function"""
    results = forecast_demand(
        sample_time_series,
        'TEST-001',
        'Test Item',
        models=['arima', 'gradient_boosting']
    )
    
    assert 'arima' in results
    assert 'gradient_boosting' in results
    
    # Check ARIMA results
    if 'error' not in results['arima']:
        assert 'forecast' in results['arima']
        assert len(results['arima']['forecast']) == 30


def test_ensemble_forecast(sample_time_series):
    """Test ensemble forecasting"""
    forecaster = DemandForecaster(
        sample_time_series,
        'TEST-001',
        'Test Item'
    )
    
    # Train multiple models
    forecaster.train_arima_model()
    forecaster.train_ml_model('gradient_boosting')
    
    # Get ensemble
    ensemble = forecaster.get_ensemble_forecast()
    
    assert len(ensemble) == 30
    assert 'forecasted_quantity' in ensemble.columns
    assert ensemble['forecasted_quantity'].min() >= 0


def test_forecast_non_negative():
    """Test that forecasts are non-negative"""
    # Create data with some zeros
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=120, freq='D'),
        'quantity': np.random.randint(0, 20, 120)
    })
    
    results = forecast_demand(data, 'TEST-001', 'Test Item', models=['arima'])
    
    if 'error' not in results['arima']:
        forecast = results['arima']['forecast']
        assert forecast['forecasted_quantity'].min() >= 0


def test_forecast_with_trend():
    """Test forecasting with clear trend"""
    # Create data with strong upward trend
    dates = pd.date_range('2024-01-01', periods=120, freq='D')
    quantities = np.linspace(5, 25, 120) + np.random.normal(0, 0.5, 120)
    
    data = pd.DataFrame({
        'date': dates,
        'quantity': quantities
    })
    
    results = forecast_demand(data, 'TEST-001', 'Test Item', models=['arima'])
    
    if 'error' not in results['arima']:
        forecast = results['arima']['forecast']
        # Forecast should generally follow upward trend
        assert forecast['forecasted_quantity'].mean() > quantities[:30].mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
