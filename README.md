# Inventory-Management-Demand-Forecasting-System
A comprehensive SaaS application for plant maintenance inventory management using machine learning-based demand forecasting to minimize stock outs and reduce excess inventory costs.
🎯 Features
Data Processing & Validation: Automated data cleaning, validation, and feature engineering
Multi-Model Forecasting: ARIMA, Prophet, and Gradient Boosting models for demand prediction
Inventory Optimization: Calculate optimal reorder points, safety stock, and EOQ
Email Notifications: Automated Outlook email alerts for reorder recommendations
Interactive Dashboard: Real-time visualization of inventory status and forecasts
Export Capabilities: Download recommendations as CSV or Excel files
📋 Requirements
Python 3.10+
Streamlit
Pandas, NumPy
Scikit-learn, Statsmodels, Prophet
Plotly, Matplotlib, Seaborn
🚀 Installation
Clone or download the project

Install dependencies:

pip install -r requirements.txt
Run the application:
streamlit run app.py
The application will open in your default web browser at http://localhost:8501.

📊 Data Format
The system expects inventory data with the following columns:

Column	Description	Type
date_received	Date when item was received	Date
item_code	Unique item identifier	String
item_description	Item description	String
received_quantity	Quantity received	Numeric
issued_quantity	Quantity issued	Numeric
stock_onhand	Current stock level	Numeric
equipment_name	Equipment using the item	String
unit_cost	Cost per unit	Numeric
issued_date	Date when item was issued	Date
item_PR_number	Purchase requisition number	String
item_PO_number	Purchase order number	String
Sample Data
A sample dataset (sample_inventory_data.csv) is included for testing and demonstration purposes.

🔧 Usage Guide
1. Data Upload
Option A: Upload your own CSV or Excel file
Option B: Use the provided sample data for testing
2. Data Processing
Navigate to the “Data Processing” tab
Review the data overview and quality metrics
Click “Process Data” to clean and prepare the data
Review the processed data summary and item summary
3. Demand Forecasting
Navigate to the “Demand Forecasting” tab
Select an item from the dropdown
Choose forecasting models (ARIMA, Prophet, Gradient Boosting)
Click “Generate Forecast” for individual items
Or click “Forecast All Items” to process all items at once
4. Inventory Analysis
Navigate to the “Inventory Analysis” tab
Set your desired service level (default: 95%)
Click “Generate Recommendations”
Review inventory metrics and reorder recommendations
Export results as CSV or Excel
5. Email Notifications
Configure email settings in the sidebar:
Sender email (Outlook)
Email password or app password
Recipient email
Test the connection
Navigate to “Email Notifications” tab
Choose batch or individual email mode
Send notifications for items requiring reorder
6. Dashboard
View comprehensive analytics including:

Key performance indicators
Inventory status overview
Top items by urgency
Stock distribution charts
Critical items requiring immediate attention
🧮 Calculation Methods
Safety Stock
Safety Stock = Z-score × Demand Std Dev × √Lead Time × Safety Multiplier
Reorder Point
Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock
Economic Order Quantity (EOQ)
EOQ = √((2 × Annual Demand × Ordering Cost) / Holding Cost)
Minimum Stock Level
MSL = Average Daily Demand × Lead Time
🔬 Forecasting Models
ARIMA (AutoRegressive Integrated Moving Average)
Best for: Time series with trends and patterns
Parameters: (p, d, q) = (1, 1, 1) default
Provides: Point forecasts with statistical metrics
Prophet
Best for: Data with strong seasonal patterns
Features: Handles holidays and missing data
Provides: Forecasts with confidence intervals
Gradient Boosting
Best for: Complex patterns with multiple features
Features: Uses lag features and rolling statistics
Provides: High accuracy for non-linear patterns
📧 Email Configuration
Outlook Setup
Use App Password (recommended):

Go to Microsoft Account Security
Enable two-factor authentication
Generate an app password
Use the app password instead of your regular password
SMTP Settings (configured automatically):

Server: smtp-mail.outlook.com
Port: 587
Security: STARTTLS
Email Templates
The system sends two types of emails:

Individual Alerts: One email per item requiring reorder
Batch Alerts: One email with all items requiring reorder (includes CSV attachment)
🧪 Testing
Run unit tests:

# Test data preprocessing
pytest test_data_preprocessing.py -v

# Test forecasting models
pytest test_forecasting_model.py -v

# Test inventory logic
pytest test_inventory_logic.py -v

# Run all tests
pytest -v
📁 Project Structure
.
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration constants
├── utils.py                        # Utility functions
├── data_preprocessing.py           # Data cleaning and feature engineering
├── forecasting_model.py            # Demand forecasting models
├── inventory_logic.py              # Inventory optimization calculations
├── email_service.py                # Email notification service
├── sample_inventory_data.csv       # Sample dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── test_data_preprocessing.py      # Unit tests for preprocessing
├── test_forecasting_model.py       # Unit tests for forecasting
└── test_inventory_logic.py         # Unit tests for inventory logic
🎨 Customization
Adjust Forecasting Parameters
Edit config.py:

FORECAST_HORIZON_DAYS = 30          # Forecast period
DEFAULT_SERVICE_LEVEL = 0.95        # Target service level
DEFAULT_LEAD_TIME_DAYS = 7          # Default lead time
ARIMA_ORDER = (1, 1, 1)            # ARIMA parameters
Modify Email Templates
Edit config.py:

EMAIL_SUBJECT_TEMPLATE = "Your custom subject"
EMAIL_BODY_TEMPLATE = """Your custom email body"""
🐛 Troubleshooting
Common Issues
Prophet Installation Error:

# On Windows, install C++ build tools first
pip install prophet --no-cache-dir
Email Authentication Failed:

Use an app password instead of your regular password
Ensure two-factor authentication is enabled
Check that SMTP access is allowed in your account
Insufficient Data Warning:

Ensure at least 90 days of historical data per item
Check for missing or invalid dates in your data
Memory Issues with Large Datasets:

Process items in batches
Reduce the number of forecasting models
Filter data by date range
📈 Performance Tips
For Large Datasets:

Use sample data for initial testing
Process items in batches
Consider using only ARIMA or Gradient Boosting (faster than Prophet)
For Better Forecasts:

Ensure at least 120 days of historical data
Remove outliers or explain them in your data
Adjust service level based on item criticality
For Production Deployment:

Use environment variables for email credentials
Implement database storage for results
Add user authentication
Schedule automated forecasting runs
🔒 Security Notes
Never commit email credentials to version control
Use environment variables or secure vaults for sensitive data
Implement proper authentication for production deployments
Regularly update dependencies for security patches
📝 License
This project is provided as-is for educational and commercial use.

🤝 Support
For issues, questions, or contributions:

Check the troubleshooting section
Review the documentation
Contact the development team
🔄 Version History
v1.0.0 (2025-01-10): Initial release
Multi-model demand forecasting
Inventory optimization
Email notifications
Interactive dashboard
🎓 References
ARIMA: Box, G. E. P., & Jenkins, G. M. (1970)
Prophet: Taylor, S. J., & Letham, B. (2018)
EOQ: Harris, F. W. (1913)
Safety Stock: Silver, E. A., Pyke, D. F., & Peterson, R. (1998)
