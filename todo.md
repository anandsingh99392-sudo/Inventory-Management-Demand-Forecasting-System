# AI-Driven Inventory Management & Demand Forecasting System

## Project Overview
A comprehensive SaaS application for plant maintenance inventory management using ML-based demand forecasting to minimize stockouts and reduce excess inventory costs.

## Technology Stack
- **Frontend**: Streamlit (Python 3.10+)
- **Data Processing**: Pandas, NumPy
- **ML/Forecasting**: Scikit-learn, Statsmodels, Prophet
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Email**: smtplib (Outlook integration)

## Module Structure

### 1. Core Modules (Python Files)
- `app.py` - Main Streamlit application entry point
- `data_preprocessing.py` - Data cleaning, validation, feature engineering
- `forecasting_model.py` - Time-series forecasting models (ARIMA, Prophet, XGBoost)
- `inventory_logic.py` - Reorder level, minimum stock level calculations
- `email_service.py` - Outlook email notification system
- `config.py` - Configuration constants and parameters
- `utils.py` - Helper functions and utilities

### 2. Test Files
- `test_data_preprocessing.py` - Unit tests for data preprocessing
- `test_forecasting_model.py` - Unit tests for forecasting models
- `test_inventory_logic.py` - Unit tests for inventory calculations

### 3. Sample Data
- `sample_inventory_data.csv` - Sample dataset for demonstration

### 4. Documentation
- `README.md` - Project documentation and setup instructions

## Key Features to Implement

### Data Preprocessing Module
- Load and validate inventory data
- Handle missing values (imputation/removal)
- Convert date columns to datetime
- Feature engineering: monthly/yearly aggregations, lead time, MSL, ROL
- Data quality checks and error handling

### Forecasting Module
- ARIMA model for time-series forecasting
- Prophet model for seasonal patterns
- XGBoost Regressor for demand prediction
- Model evaluation metrics (RMSE, MAE, MAPE)
- 30-day demand forecast generation

### Inventory Logic Module
- Calculate optimal reorder levels
- Determine minimum stock levels
- Service level calculations (95% default)
- Reorder point recommendations
- Safety stock calculations

### Email Service Module
- SMTP configuration for Outlook
- Email templates for reorder alerts
- Trigger notifications when stock < reorder level
- Batch email sending for multiple items

### Streamlit Dashboard
- File upload interface for inventory data
- Data preview and statistics
- Interactive demand forecast visualizations
- Inventory status dashboard
- Reorder recommendations table
- Email configuration and testing
- Export results to CSV/Excel

## Development Order
1. Create configuration file with constants
2. Implement data preprocessing module with error handling
3. Build forecasting models (ARIMA, Prophet, XGBoost)
4. Develop inventory logic calculations
5. Create email service integration
6. Build Streamlit UI with all features
7. Generate sample data for testing
8. Write unit tests for all modules
9. Create comprehensive README documentation

## Design Decisions
- **Modular Architecture**: Separate concerns for maintainability
- **Multiple Models**: Offer ARIMA, Prophet, XGBoost for flexibility
- **Robust Error Handling**: Validate inputs, handle edge cases
- **Interactive UI**: Streamlit for easy deployment and usage
- **Email Integration**: Automated alerts for proactive inventory management
- **Sample Data**: Included for immediate testing and demonstration
