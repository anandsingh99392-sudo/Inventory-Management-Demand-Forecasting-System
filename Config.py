"""
Configuration file for AI-Driven Inventory Management System
Contains all constants, parameters, and configuration settings
"""
# Data Schema Configuration
REQUIRED_COLUMNS = [
    'date_received',
    'item_code',
    'item_description',
    'received_quantity',
    'issued_quantity',
    'stock_onhand',
    'equipment_name',
    'unit_cost',
    'issued_date',
    'item_PR_number',
    'item_PO_number'
]
# Date columns that need datetime conversion
DATE_COLUMNS = ['date_received', 'issued_date']

# Forecasting Parameters
FORECAST_HORIZON_DAYS = 30  # Default forecast period
MIN_HISTORICAL_DAYS = 90  # Minimum data required for forecasting

# Inventory Management Parameters
DEFAULT_SERVICE_LEVEL = 0.95  # 95% service level
DEFAULT_LEAD_TIME_DAYS = 7  # Default lead time if not specified
SAFETY_STOCK_MULTIPLIER = 1.5  # Safety stock factor

# Model Parameters
ARIMA_ORDER = (1, 1, 1)  # Default ARIMA (p, d, q) parameters
PROPHET_SEASONALITY_MODE = 'multiplicative'
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
}

# Email Configuration (to be set by user)
SMTP_SERVER = 'smtp-mail.outlook.com'
SMTP_PORT = 587
EMAIL_SUBJECT_TEMPLATE = "Inventory Reorder Alert: {item_description}"
EMAIL_BODY_TEMPLATE = """
Dear Inventory Manager,

This is an automated alert for the following item:

Item Code: {item_code}
Item Description: {item_description}
Equipment: {equipment_name}
Current Stock: {current_stock}
Reorder Level: {reorder_level}
Recommended Order Quantity: {order_quantity}
Forecasted Demand (30 days): {forecasted_demand}

Action Required: Please initiate purchase order to replenish stock.
Best regards,
AI Inventory Management System
"""
# Visualization Settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 6)
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Data Quality Thresholds
MAX_MISSING_PERCENTAGE = 0.3  # Maximum 30% missing values allowed
MIN_DATA_POINTS = 30  # Minimum data points required per item

# File Upload Settings
MAX_FILE_SIZE_MB = 50
ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']
