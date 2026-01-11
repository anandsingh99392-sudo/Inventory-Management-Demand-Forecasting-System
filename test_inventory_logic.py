"""
Unit tests for inventory logic module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from inventory_logic import (
    InventoryOptimizer,
    generate_reorder_recommendations,
    calculate_inventory_value
)


@pytest.fixture
def sample_item_data():
    """Create sample item data"""
    return {
        'item_code': 'TEST-001',
        'item_description': 'Test Item',
        'equipment_name': 'Test Equipment',
        'stock_onhand': 50,
        'unit_cost': 25.50,
        'lead_time_days': 7
    }


@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data"""
    dates = pd.date_range(start='2025-01-11', periods=30, freq='D')
    quantities = np.random.randint(3, 10, 30)
    
    return pd.DataFrame({
        'date': dates,
        'forecasted_quantity': quantities
    })


def test_optimizer_initialization(sample_item_data, sample_forecast_data):
    """Test optimizer initialization"""
    optimizer = InventoryOptimizer(
        sample_item_data,
        sample_forecast_data,
        service_level=0.95
    )
    
    assert optimizer.service_level == 0.95
    assert optimizer.z_score > 0


def test_calculate_safety_stock(sample_item_data, sample_forecast_data):
    """Test safety stock calculation"""
    optimizer = InventoryOptimizer(
        sample_item_data,
        sample_forecast_data,
        service_level=0.95
    )
    
    avg_demand = 5.0
    demand_std = 2.0
    lead_time = 7.0
    
    safety_stock = optimizer.calculate_safety_stock(avg_demand, demand_std, lead_time)
    
    assert safety_stock >= 0
    assert isinstance(safety_stock, float)


def test_calculate_reorder_point(sample_item_data, sample_forecast_data):
    """Test reorder point calculation"""
    optimizer = InventoryOptimizer(
        sample_item_data,
        sample_forecast_data,
        service_level=0.95
    )
    
    avg_demand = 5.0
    lead_time = 7.0
    safety_stock = 10.0
    
    reorder_point = optimizer.calculate_reorder_point(avg_demand, lead_time, safety_stock)
    
    assert reorder_point >= 0
    assert reorder_point >= safety_stock


def test_calculate_eoq(sample_item_data, sample_forecast_data):
    """Test Economic Order Quantity calculation"""
    optimizer = InventoryOptimizer(
        sample_item_data,
        sample_forecast_data,
        service_level=0.95
    )
    
    annual_demand = 1000.0
    eoq = optimizer.calculate_economic_order_quantity(annual_demand)
    
    assert eoq >= 0
    assert isinstance(eoq, float)


def test_analyze_item_needs_reorder(sample_item_data, sample_forecast_data):
    """Test item analysis when reorder is needed"""
    optimizer = InventoryOptimizer(
        sample_item_data,
        sample_forecast_data,
        service_level=0.95
    )
    
    result = optimizer.analyze_item('TEST-001', current_stock=10, lead_time=7)
    
    assert 'item_code' in result
    assert 'needs_reorder' in result
    assert 'recommended_order_quantity' in result


def test_generate_reorder_recommendations():
    """Test generating recommendations for multiple items"""
    item_summary = pd.DataFrame({
        'item_code': ['ITEM-001', 'ITEM-002', 'ITEM-003'],
        'item_description': ['Item 1', 'Item 2', 'Item 3'],
        'equipment_name': ['Equip A', 'Equip B', 'Equip C'],
        'stock_onhand': [10, 100, 50],
        'unit_cost': [25.50, 15.75, 35.00],
        'lead_time_days': [7, 5, 10]
    })
    
    forecast_results = {}
    for item_code in ['ITEM-001', 'ITEM-002', 'ITEM-003']:
        dates = pd.date_range(start='2025-01-11', periods=30, freq='D')
        quantities = np.random.randint(3, 10, 30)
        
        forecast_results[item_code] = {
            'arima': {
                'forecast': pd.DataFrame({
                    'date': dates,
                    'forecasted_quantity': quantities
                })
            }
        }
    
    recommendations = generate_reorder_recommendations(
        item_summary,
        forecast_results,
        service_level=0.95
    )
    
    assert len(recommendations) > 0
    assert 'item_code' in recommendations.columns
    assert 'needs_reorder' in recommendations.columns


def test_calculate_inventory_value():
    """Test inventory value calculation"""
    recommendations = pd.DataFrame({
        'item_code': ['ITEM-001', 'ITEM-002', 'ITEM-003'],
        'current_stock': [50, 100, 75],
        'unit_cost': [25.50, 15.75, 35.00],
        'recommended_order_quantity': [20, 0, 30],
        'needs_reorder': [True, False, True],
        'inventory_turnover_ratio': [2.5, 3.0, 1.8]
    })
    
    metrics = calculate_inventory_value(recommendations)
    
    assert 'total_current_inventory_value' in metrics
    assert 'items_needing_reorder' in metrics
    assert metrics['total_items'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
