"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_preprocessing import (
    InventoryDataPreprocessor,
    preprocess_inventory_data
)


@pytest.fixture
def sample_data():
    """Create sample inventory data for testing"""
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    
    data = {
        'date_received': dates,
        'item_code': ['ITEM-001'] * 30,
        'item_description': ['Test Item'] * 30,
        'received_quantity': [10, 0, 0, 20, 0] * 6,
        'issued_quantity': [2, 3, 2, 4, 3] * 6,
        'stock_onhand': [100 - i*2 for i in range(30)],
        'equipment_name': ['Equipment A'] * 30,
        'unit_cost': [25.50] * 30,
        'issued_date': dates + timedelta(days=1),
        'item_PR_number': ['PR-001'] * 30,
        'item_PO_number': ['PO-001'] * 30
    }
    
    return pd.DataFrame(data)


def test_preprocessor_initialization(sample_data):
    """Test preprocessor initialization"""
    preprocessor = InventoryDataPreprocessor(sample_data)
    
    assert preprocessor.raw_data is not None
    assert len(preprocessor.raw_data) == 30
    assert preprocessor.processed_data is None


def test_data_validation(sample_data):
    """Test data validation"""
    preprocessor = InventoryDataPreprocessor(sample_data)
    validation = preprocessor.validate_data()
    
    assert validation['is_valid'] == True
    assert 'errors' in validation
    assert 'warnings' in validation


def test_data_validation_empty():
    """Test validation with empty DataFrame"""
    empty_df = pd.DataFrame()
    preprocessor = InventoryDataPreprocessor(empty_df)
    validation = preprocessor.validate_data()
    
    assert validation['is_valid'] == False
    assert len(validation['errors']) > 0


def test_clean_data(sample_data):
    """Test data cleaning"""
    preprocessor = InventoryDataPreprocessor(sample_data)
    cleaned = preprocessor.clean_data()
    
    assert len(cleaned) > 0
    assert 'date_received' in cleaned.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned['date_received'])


def test_clean_data_with_missing_values():
    """Test cleaning data with missing values"""
    data = pd.DataFrame({
        'date_received': ['2025-01-01', '2025-01-02', None],
        'item_code': ['ITEM-001', 'ITEM-002', 'ITEM-003'],
        'item_description': ['Item 1', None, 'Item 3'],
        'received_quantity': [10, None, 15],
        'issued_quantity': [2, 3, None],
        'stock_onhand': [100, 95, 90],
        'equipment_name': ['Equip A', 'Equip B', None],
        'unit_cost': [25.50, None, 30.00],
        'issued_date': ['2025-01-02', '2025-01-03', '2025-01-04'],
        'item_PR_number': ['PR-001', 'PR-002', 'PR-003'],
        'item_PO_number': ['PO-001', 'PO-002', 'PO-003']
    })
    
    preprocessor = InventoryDataPreprocessor(data)
    cleaned = preprocessor.clean_data()
    
    # Should remove rows with invalid dates
    assert len(cleaned) < len(data)
    
    # Missing quantities should be filled with 0
    assert cleaned['received_quantity'].isnull().sum() == 0
    assert cleaned['issued_quantity'].isnull().sum() == 0


def test_engineer_features(sample_data):
    """Test feature engineering"""
    preprocessor = InventoryDataPreprocessor(sample_data)
    cleaned = preprocessor.clean_data()
    engineered = preprocessor.engineer_features(cleaned)
    
    # Check for new features
    assert 'year' in engineered.columns
    assert 'month' in engineered.columns
    assert 'lead_time_days' in engineered.columns
    assert 'total_received' in engineered.columns
    assert 'total_issued' in engineered.columns


def test_aggregate_by_item(sample_data):
    """Test item aggregation"""
    preprocessor = InventoryDataPreprocessor(sample_data)
    cleaned = preprocessor.clean_data()
    engineered = preprocessor.engineer_features(cleaned)
    aggregated = preprocessor.aggregate_by_item(engineered)
    
    assert len(aggregated) == 1  # Only one unique item
    assert 'item_code' in aggregated.columns
    assert 'total_transactions' in aggregated.columns


def test_prepare_time_series(sample_data):
    """Test time series preparation"""
    preprocessor = InventoryDataPreprocessor(sample_data)
    cleaned = preprocessor.clean_data()
    
    ts_data = preprocessor.prepare_time_series(cleaned, 'ITEM-001')
    
    assert len(ts_data) > 0
    assert 'date' in ts_data.columns
    assert 'quantity' in ts_data.columns


def test_full_preprocessing_pipeline(sample_data):
    """Test complete preprocessing pipeline"""
    processed_data, item_summary, validation_report = preprocess_inventory_data(sample_data)
    
    assert processed_data is not None
    assert item_summary is not None
    assert validation_report['is_valid'] == True
    assert len(item_summary) > 0


def test_preprocessing_with_multiple_items():
    """Test preprocessing with multiple items"""
    data = pd.DataFrame({
        'date_received': pd.date_range('2025-01-01', periods=60, freq='D'),
        'item_code': ['ITEM-001'] * 30 + ['ITEM-002'] * 30,
        'item_description': ['Item 1'] * 30 + ['Item 2'] * 30,
        'received_quantity': [10] * 60,
        'issued_quantity': [2] * 60,
        'stock_onhand': list(range(100, 40, -1)) + list(range(100, 40, -1)),
        'equipment_name': ['Equipment A'] * 60,
        'unit_cost': [25.50] * 60,
        'issued_date': pd.date_range('2025-01-02', periods=60, freq='D'),
        'item_PR_number': ['PR-001'] * 60,
        'item_PO_number': ['PO-001'] * 60
    })
    
    processed_data, item_summary, validation_report = preprocess_inventory_data(data)
    
    assert len(item_summary) == 2  # Two unique items
    assert 'ITEM-001' in item_summary['item_code'].values
    assert 'ITEM-002' in item_summary['item_code'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
