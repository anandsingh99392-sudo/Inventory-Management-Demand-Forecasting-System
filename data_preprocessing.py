"""
Data Preprocessing Module
Handles data cleaning, validation, and feature engineering for inventory data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from config import (
    REQUIRED_COLUMNS, DATE_COLUMNS, MAX_MISSING_PERCENTAGE, 
    MIN_DATA_POINTS, DEFAULT_LEAD_TIME_DAYS
)
from utils import (
    validate_dataframe_schema, calculate_date_range, 
    generate_date_features, calculate_moving_average
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryDataPreprocessor:
    """
    Main class for preprocessing inventory data
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor with inventory data
        
        Args:
            df: Raw inventory DataFrame
        """
        self.raw_data = df.copy()
        self.processed_data = None
        self.validation_report = {}
        self.preprocessing_log = []
        
    def validate_data(self) -> Dict:
        """
        Validate the input data structure and quality
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting data validation...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Check if DataFrame is empty
        if self.raw_data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame is empty")
            return validation_results
        
        # Validate schema
        schema_validation = validate_dataframe_schema(self.raw_data, REQUIRED_COLUMNS)
        if not schema_validation['is_valid']:
            validation_results['is_valid'] = False
            validation_results['errors'].append(schema_validation['message'])
        
        # Check for missing values
        missing_pct = (self.raw_data.isnull().sum() / len(self.raw_data)) * 100
        high_missing = missing_pct[missing_pct > (MAX_MISSING_PERCENTAGE * 100)]
        
        if not high_missing.empty:
            validation_results['warnings'].append(
                f"High missing values in columns: {high_missing.to_dict()}"
            )
        
        # Check data types
        validation_results['info']['shape'] = self.raw_data.shape
        validation_results['info']['columns'] = list(self.raw_data.columns)
        validation_results['info']['missing_values'] = missing_pct.to_dict()
        
        self.validation_report = validation_results
        logger.info(f"Validation complete. Valid: {validation_results['is_valid']}")
        
        return validation_results
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the inventory data by handling missing values and data quality issues
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df = self.raw_data.copy()
        
        # Convert date columns to datetime
        for date_col in DATE_COLUMNS:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                invalid_dates = df[date_col].isnull().sum()
                if invalid_dates > 0:
                    logger.warning(f"Found {invalid_dates} invalid dates in {date_col}")
        
        # Handle missing values in numeric columns
        numeric_columns = ['received_quantity', 'issued_quantity', 'stock_onhand', 'unit_cost']
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, coercing errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill missing values with 0 for quantities
                if 'quantity' in col.lower() or 'stock' in col.lower():
                    df[col] = df[col].fillna(0)
                # Fill missing unit_cost with median
                elif 'cost' in col.lower():
                    median_cost = df[col].median()
                    df[col] = df[col].fillna(median_cost)
        
        # Handle missing values in categorical columns
        categorical_columns = ['item_code', 'item_description', 'equipment_name', 
                               'item_PR_number', 'item_PO_number']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('UNKNOWN')
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows with invalid dates in critical columns
        if 'date_received' in df.columns:
            df = df.dropna(subset=['date_received'])
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Ensure non-negative quantities
        for col in ['received_quantity', 'issued_quantity', 'stock_onhand']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        self.preprocessing_log.append(f"Cleaned data: {len(df)} rows remaining")
        logger.info(f"Data cleaning complete. Rows: {len(df)}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for analysis and modeling
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        df = df.copy()
        
        # Generate date-based features
        if 'date_received' in df.columns:
            df = generate_date_features(df, 'date_received')
        
        if 'issued_date' in df.columns:
            df = generate_date_features(df, 'issued_date')
            # Rename to avoid conflicts
            df = df.rename(columns={
                'year': 'issued_year',
                'month': 'issued_month',
                'quarter': 'issued_quarter'
            })
        
        # Calculate inventory metrics per item
        if 'item_code' in df.columns:
            # Total received and issued per item
            df['total_received'] = df.groupby('item_code')['received_quantity'].transform('sum')
            df['total_issued'] = df.groupby('item_code')['issued_quantity'].transform('sum')
            
            # Average stock level per item
            df['avg_stock'] = df.groupby('item_code')['stock_onhand'].transform('mean')
            
            # Inventory turnover rate (issued / average stock)
            df['turnover_rate'] = df.apply(
                lambda row: row['total_issued'] / row['avg_stock'] if row['avg_stock'] > 0 else 0,
                axis=1
            )
        
        # Calculate lead time (days between received and issued)
        if 'date_received' in df.columns and 'issued_date' in df.columns:
            df['lead_time_days'] = (df['issued_date'] - df['date_received']).dt.days
            df['lead_time_days'] = df['lead_time_days'].clip(lower=0)
            df['lead_time_days'] = df['lead_time_days'].fillna(DEFAULT_LEAD_TIME_DAYS)
        else:
            df['lead_time_days'] = DEFAULT_LEAD_TIME_DAYS
        
        # Calculate total value
        if 'stock_onhand' in df.columns and 'unit_cost' in df.columns:
            df['total_value'] = df['stock_onhand'] * df['unit_cost']
        
        # Days since last received
        if 'date_received' in df.columns:
            max_date = df['date_received'].max()
            df['days_since_received'] = (max_date - df['date_received']).dt.days
        
        # Moving averages for issued quantities (if enough data)
        if 'issued_quantity' in df.columns and len(df) > 7:
            df = df.sort_values('issued_date') if 'issued_date' in df.columns else df
            df['issued_ma_7'] = calculate_moving_average(df['issued_quantity'], window=7)
            df['issued_ma_30'] = calculate_moving_average(df['issued_quantity'], window=30)
        
        self.preprocessing_log.append(f"Feature engineering complete. Features: {len(df.columns)}")
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def aggregate_by_item(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by item code for inventory analysis
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Aggregated DataFrame by item
        """
        logger.info("Aggregating data by item...")
        
        agg_dict = {
            'item_description': 'first',
            'equipment_name': 'first',
            'received_quantity': 'sum',
            'issued_quantity': 'sum',
            'stock_onhand': 'last',
            'unit_cost': 'mean',
            'total_value': 'last',
            'lead_time_days': 'mean',
            'turnover_rate': 'mean'
        }
        
        # Only include columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        item_summary = df.groupby('item_code').agg(agg_dict).reset_index()
        
        # Calculate additional metrics
        item_summary['total_transactions'] = df.groupby('item_code').size().values
        
        logger.info(f"Aggregated to {len(item_summary)} unique items")
        
        return item_summary
    
    def prepare_time_series(self, df: pd.DataFrame, item_code: str, 
                           date_column: str = 'issued_date',
                           value_column: str = 'issued_quantity') -> pd.DataFrame:
        """
        Prepare time series data for a specific item
        
        Args:
            df: Processed DataFrame
            item_code: Item code to filter
            date_column: Date column name
            value_column: Value column name
            
        Returns:
            Time series DataFrame for the item
        """
        logger.info(f"Preparing time series for item: {item_code}")
        
        # Filter for specific item
        item_data = df[df['item_code'] == item_code].copy()
        
        if item_data.empty:
            logger.warning(f"No data found for item: {item_code}")
            return pd.DataFrame()
        
        # Ensure date column is datetime
        item_data[date_column] = pd.to_datetime(item_data[date_column])
        
        # Sort by date
        item_data = item_data.sort_values(date_column)
        
        # Create daily time series
        item_data = item_data.set_index(date_column)
        
        # Resample to daily frequency and sum quantities
        ts_data = item_data[value_column].resample('D').sum()
        
        # Fill missing dates with 0
        ts_data = ts_data.fillna(0)
        
        # Convert back to DataFrame
        ts_df = ts_data.reset_index()
        ts_df.columns = ['date', 'quantity']
        
        logger.info(f"Time series prepared: {len(ts_df)} data points")
        
        return ts_df
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full preprocessing pipeline
        
        Returns:
            Tuple of (processed_data, item_summary)
        """
        logger.info("Starting full preprocessing pipeline...")
        
        # Validate
        validation = self.validate_data()
        if not validation['is_valid']:
            raise ValueError(f"Data validation failed: {validation['errors']}")
        
        # Clean
        cleaned_data = self.clean_data()
        
        # Engineer features
        processed_data = self.engineer_features(cleaned_data)
        
        # Aggregate by item
        item_summary = self.aggregate_by_item(processed_data)
        
        self.processed_data = processed_data
        
        logger.info("Preprocessing pipeline complete!")
        
        return processed_data, item_summary


def preprocess_inventory_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main function to preprocess inventory data
    
    Args:
        df: Raw inventory DataFrame
        
    Returns:
        Tuple of (processed_data, item_summary, validation_report)
    """
    preprocessor = InventoryDataPreprocessor(df)
    processed_data, item_summary = preprocessor.process()
    
    return processed_data, item_summary, preprocessor.validation_report
