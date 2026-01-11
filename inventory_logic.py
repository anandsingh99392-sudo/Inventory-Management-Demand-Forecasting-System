"""
Inventory Management Logic Module
Calculates optimal reorder levels, safety stock, and generates recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from datetime import datetime, timedelta

from config import (
    DEFAULT_SERVICE_LEVEL, DEFAULT_LEAD_TIME_DAYS, 
    SAFETY_STOCK_MULTIPLIER, FORECAST_HORIZON_DAYS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """
    Main class for inventory optimization calculations
    """
    
    def __init__(self, item_data: pd.DataFrame, forecast_data: pd.DataFrame, 
                 service_level: float = DEFAULT_SERVICE_LEVEL):
        """
        Initialize inventory optimizer
        
        Args:
            item_data: DataFrame with item information
            forecast_data: DataFrame with demand forecast
            service_level: Target service level (0-1)
        """
        self.item_data = item_data
        self.forecast_data = forecast_data
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)
        
    def calculate_safety_stock(self, avg_demand: float, demand_std: float, 
                               lead_time: float) -> float:
        """
        Calculate safety stock level
        
        Args:
            avg_demand: Average daily demand
            demand_std: Standard deviation of demand
            lead_time: Lead time in days
            
        Returns:
            Safety stock quantity
        """
        try:
            # Safety stock = Z-score * std_dev * sqrt(lead_time)
            safety_stock = self.z_score * demand_std * np.sqrt(lead_time)
            
            # Apply multiplier for additional buffer
            safety_stock *= SAFETY_STOCK_MULTIPLIER
            
            return max(0, safety_stock)
            
        except Exception as e:
            logger.error(f"Error calculating safety stock: {str(e)}")
            return 0
    
    def calculate_reorder_point(self, avg_demand: float, lead_time: float, 
                                safety_stock: float) -> float:
        """
        Calculate reorder point
        
        Args:
            avg_demand: Average daily demand
            lead_time: Lead time in days
            safety_stock: Safety stock quantity
            
        Returns:
            Reorder point quantity
        """
        try:
            # Reorder Point = (Average Daily Demand * Lead Time) + Safety Stock
            reorder_point = (avg_demand * lead_time) + safety_stock
            
            return max(0, reorder_point)
            
        except Exception as e:
            logger.error(f"Error calculating reorder point: {str(e)}")
            return 0
    
    def calculate_economic_order_quantity(self, annual_demand: float, 
                                         ordering_cost: float = 100, 
                                         holding_cost_rate: float = 0.25) -> float:
        """
        Calculate Economic Order Quantity (EOQ)
        
        Args:
            annual_demand: Annual demand quantity
            ordering_cost: Cost per order
            holding_cost_rate: Annual holding cost as percentage of unit cost
            
        Returns:
            Optimal order quantity
        """
        try:
            if annual_demand <= 0:
                return 0
            
            # Get unit cost from item data
            unit_cost = self.item_data.get('unit_cost', 100)
            holding_cost = unit_cost * holding_cost_rate
            
            # EOQ = sqrt((2 * D * S) / H)
            # D = annual demand, S = ordering cost, H = holding cost
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            
            return max(0, eoq)
            
        except Exception as e:
            logger.error(f"Error calculating EOQ: {str(e)}")
            return 0
    
    def calculate_minimum_stock_level(self, avg_demand: float, 
                                     lead_time: float) -> float:
        """
        Calculate minimum stock level (MSL)
        
        Args:
            avg_demand: Average daily demand
            lead_time: Lead time in days
            
        Returns:
            Minimum stock level
        """
        try:
            # MSL = Average Daily Demand * Lead Time
            msl = avg_demand * lead_time
            
            return max(0, msl)
            
        except Exception as e:
            logger.error(f"Error calculating MSL: {str(e)}")
            return 0
    
    def calculate_maximum_stock_level(self, reorder_level: float, 
                                     eoq: float) -> float:
        """
        Calculate maximum stock level
        
        Args:
            reorder_level: Reorder level quantity
            eoq: Economic order quantity
            
        Returns:
            Maximum stock level
        """
        try:
            # Maximum Stock Level = Reorder Level + EOQ
            max_stock = reorder_level + eoq
            
            return max(0, max_stock)
            
        except Exception as e:
            logger.error(f"Error calculating maximum stock level: {str(e)}")
            return 0
    
    def analyze_item(self, item_code: str, current_stock: float, 
                    lead_time: float = None) -> Dict:
        """
        Perform complete inventory analysis for an item
        
        Args:
            item_code: Item code
            current_stock: Current stock on hand
            lead_time: Lead time in days (optional)
            
        Returns:
            Dictionary with inventory recommendations
        """
        logger.info(f"Analyzing inventory for item: {item_code}")
        
        try:
            # Get forecast data for this item
            if self.forecast_data.empty:
                return {'error': 'No forecast data available'}
            
            forecast = self.forecast_data['forecasted_quantity'].values
            
            # Calculate demand statistics
            avg_demand = np.mean(forecast)
            demand_std = np.std(forecast)
            total_forecast = np.sum(forecast)
            
            # Use provided lead time or default
            if lead_time is None:
                lead_time = self.item_data.get('lead_time_days', DEFAULT_LEAD_TIME_DAYS)
            
            # Calculate inventory metrics
            safety_stock = self.calculate_safety_stock(avg_demand, demand_std, lead_time)
            reorder_point = self.calculate_reorder_point(avg_demand, lead_time, safety_stock)
            msl = self.calculate_minimum_stock_level(avg_demand, lead_time)
            
            # Calculate annual demand (extrapolate from forecast)
            annual_demand = (total_forecast / FORECAST_HORIZON_DAYS) * 365
            eoq = self.calculate_economic_order_quantity(annual_demand)
            max_stock = self.calculate_maximum_stock_level(reorder_point, eoq)
            
            # Determine if reorder is needed
            needs_reorder = current_stock < reorder_point
            
            # Calculate recommended order quantity
            if needs_reorder:
                order_quantity = max(eoq, reorder_point - current_stock)
            else:
                order_quantity = 0
            
            # Calculate days until stockout (if no reorder)
            if avg_demand > 0:
                days_until_stockout = current_stock / avg_demand
            else:
                days_until_stockout = float('inf')
            
            # Calculate inventory turnover
            if current_stock > 0:
                turnover_ratio = annual_demand / current_stock
            else:
                turnover_ratio = 0
            
            result = {
                'item_code': item_code,
                'current_stock': current_stock,
                'avg_daily_demand': avg_demand,
                'demand_std': demand_std,
                'forecasted_30day_demand': total_forecast,
                'annual_demand_estimate': annual_demand,
                'lead_time_days': lead_time,
                'safety_stock': safety_stock,
                'minimum_stock_level': msl,
                'reorder_point': reorder_point,
                'economic_order_quantity': eoq,
                'maximum_stock_level': max_stock,
                'needs_reorder': needs_reorder,
                'recommended_order_quantity': order_quantity,
                'days_until_stockout': days_until_stockout,
                'inventory_turnover_ratio': turnover_ratio,
                'service_level': self.service_level,
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            logger.info(f"Analysis complete for {item_code}. Reorder needed: {needs_reorder}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing item {item_code}: {str(e)}")
            return {'error': str(e), 'item_code': item_code}


def generate_reorder_recommendations(item_summary: pd.DataFrame, 
                                    forecast_results: Dict,
                                    service_level: float = DEFAULT_SERVICE_LEVEL) -> pd.DataFrame:
    """
    Generate reorder recommendations for all items
    
    Args:
        item_summary: DataFrame with item information
        forecast_results: Dictionary with forecast data for each item
        service_level: Target service level
        
    Returns:
        DataFrame with reorder recommendations
    """
    logger.info("Generating reorder recommendations for all items...")
    
    recommendations = []
    
    for _, item in item_summary.iterrows():
        item_code = item['item_code']
        
        # Get forecast for this item
        if item_code not in forecast_results:
            logger.warning(f"No forecast available for item: {item_code}")
            continue
        
        forecast_data = forecast_results[item_code]
        
        # Skip if forecast has error
        if 'error' in forecast_data:
            continue
        
        # Get the best forecast (ensemble if available, otherwise first available)
        if 'ensemble' in forecast_data and 'forecast' in forecast_data['ensemble']:
            forecast_df = forecast_data['ensemble']['forecast']
        else:
            # Get first available forecast
            for model_name in ['arima', 'prophet', 'gradient_boosting']:
                if model_name in forecast_data and 'forecast' in forecast_data[model_name]:
                    forecast_df = forecast_data[model_name]['forecast']
                    break
            else:
                continue
        
        # Create optimizer
        optimizer = InventoryOptimizer(
            item_data=item.to_dict(),
            forecast_data=forecast_df,
            service_level=service_level
        )
        
        # Analyze item
        analysis = optimizer.analyze_item(
            item_code=item_code,
            current_stock=item.get('stock_onhand', 0),
            lead_time=item.get('lead_time_days', None)
        )
        
        if 'error' not in analysis:
            # Add item details
            analysis['item_description'] = item.get('item_description', '')
            analysis['equipment_name'] = item.get('equipment_name', '')
            analysis['unit_cost'] = item.get('unit_cost', 0)
            
            recommendations.append(analysis)
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    
    # Sort by urgency (days until stockout)
    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values('days_until_stockout')
    
    logger.info(f"Generated {len(recommendations_df)} recommendations")
    
    return recommendations_df


def calculate_inventory_value(recommendations_df: pd.DataFrame) -> Dict:
    """
    Calculate total inventory value and metrics
    
    Args:
        recommendations_df: DataFrame with inventory recommendations
        
    Returns:
        Dictionary with inventory value metrics
    """
    if recommendations_df.empty:
        return {}
    
    total_current_value = (recommendations_df['current_stock'] * 
                          recommendations_df['unit_cost']).sum()
    
    total_reorder_value = (recommendations_df['recommended_order_quantity'] * 
                          recommendations_df['unit_cost']).sum()
    
    items_needing_reorder = recommendations_df['needs_reorder'].sum()
    
    avg_turnover = recommendations_df['inventory_turnover_ratio'].mean()
    
    return {
        'total_current_inventory_value': total_current_value,
        'total_reorder_value': total_reorder_value,
        'items_needing_reorder': int(items_needing_reorder),
        'total_items': len(recommendations_df),
        'average_turnover_ratio': avg_turnover,
        'reorder_percentage': (items_needing_reorder / len(recommendations_df)) * 100
    }
