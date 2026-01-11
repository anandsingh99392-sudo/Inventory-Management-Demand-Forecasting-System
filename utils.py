"""
Utility functions for the inventory management system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate if DataFrame contains all required columns
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'is_valid': True,
        'missing_columns': [],
        'extra_columns': [],
        'message': 'Schema validation passed'
    }
    
    df_columns = set(df.columns)
    required_set = set(required_columns)
    
    missing = required_set - df_columns
    if missing:
        result['is_valid'] = False
        result['missing_columns'] = list(missing)
        result['message'] = f"Missing required columns: {missing}"
    
    result['extra_columns'] = list(df_columns - required_set)
    
    return result


def calculate_date_range(df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
    """
    Calculate date range statistics from a DataFrame
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        Dictionary with date range information
    """
    try:
        dates = pd.to_datetime(df[date_column], errors='coerce')
        dates = dates.dropna()
        
        if len(dates) == 0:
            return {
                'min_date': None,
                'max_date': None,
                'total_days': 0,
                'data_points': 0
            }
        
        return {
            'min_date': dates.min(),
            'max_date': dates.max(),
            'total_days': (dates.max() - dates.min()).days,
            'data_points': len(dates)
        }
    except Exception as e:
        logger.error(f"Error calculating date range: {str(e)}")
        return {
            'min_date': None,
            'max_date': None,
            'total_days': 0,
            'data_points': 0,
            'error': str(e)
        }


def calculate_statistics(series: pd.Series) -> Dict[str, float]:
    """
    Calculate descriptive statistics for a numeric series
    
    Args:
        series: Pandas Series
        
    Returns:
        Dictionary with statistical measures
    """
    try:
        return {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75))
        }
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        return {}


def generate_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Generate time-based features from a date column
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with additional date features
    """
    try:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        df['is_month_start'] = df[date_column].dt.is_month_start
        df['is_month_end'] = df[date_column].dt.is_month_end
        
        return df
    except Exception as e:
        logger.error(f"Error generating date features: {str(e)}")
        return df


def format_currency(amount: float, currency_symbol: str = '$') -> str:
    """
    Format a number as currency
    
    Args:
        amount: Numeric amount
        currency_symbol: Currency symbol to use
        
    Returns:
        Formatted currency string
    """
    try:
        return f"{currency_symbol}{amount:,.2f}"
    except:
        return f"{currency_symbol}0.00"


def calculate_moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Calculate moving average for a series
    
    Args:
        series: Input series
        window: Window size for moving average
        
    Returns:
        Series with moving average values
    """
    try:
        return series.rolling(window=window, min_periods=1).mean()
    except Exception as e:
        logger.error(f"Error calculating moving average: {str(e)}")
        return series


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        series: Input series
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    try:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return pd.Series([False] * len(series))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division fails
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default


def create_time_series(df: pd.DataFrame, date_column: str, value_column: str, 
                       freq: str = 'D') -> pd.Series:
    """
    Create a time series with specified frequency
    
    Args:
        df: Input DataFrame
        date_column: Name of date column
        value_column: Name of value column
        freq: Frequency string ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        Time series with specified frequency
    """
    try:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        
        # Resample and aggregate
        ts = df[value_column].resample(freq).sum()
        
        # Fill missing dates with 0
        ts = ts.fillna(0)
        
        return ts
    except Exception as e:
        logger.error(f"Error creating time series: {str(e)}")
        return pd.Series()
