"""
AI-Driven Inventory Management & Demand Forecasting System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import sys

# Import custom modules
from data_preprocessing import preprocess_inventory_data, InventoryDataPreprocessor
from forecasting_model import forecast_demand, DemandForecaster
from inventory_logic import (
    generate_reorder_recommendations, 
    calculate_inventory_value,
    InventoryOptimizer
)
from email_service import send_reorder_notifications, EmailService
from config import (
    FORECAST_HORIZON_DAYS, DEFAULT_SERVICE_LEVEL,
    MIN_HISTORICAL_DAYS
)
from utils import format_currency

# Page configuration
st.set_page_config(
    page_title="AI Inventory Management System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'item_summary' not in st.session_state:
        st.session_state.item_summary = None
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'email_configured' not in st.session_state:
        st.session_state.email_configured = False


def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def display_data_overview(df):
    """Display data overview and statistics"""
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Items", f"{df['item_code'].nunique():,}")
    with col3:
        st.metric("Equipment Types", f"{df['equipment_name'].nunique():,}")
    with col4:
        total_value = (df['stock_onhand'] * df['unit_cost']).sum()
        st.metric("Total Inventory Value", format_currency(total_value))
    
    # Data preview
    with st.expander("📋 View Raw Data (First 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)
    
    # Data quality metrics
    with st.expander("🔍 Data Quality Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values:**")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Percentage': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        with col2:
            st.write("**Data Types:**")
            dtypes_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            st.dataframe(dtypes_df, use_container_width=True)


def plot_demand_forecast(item_code, historical_data, forecast_data):
    """Plot historical demand and forecast"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['quantity'],
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['forecasted_quantity'],
        mode='lines+markers',
        name='Forecasted Demand',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add confidence interval if available
    if 'lower_bound' in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['lower_bound'],
            mode='lines',
            name='Confidence Interval',
            line=dict(width=0),
            fillcolor='rgba(255, 127, 14, 0.2)',
            fill='tonexty'
        ))
    
    fig.update_layout(
        title=f'Demand Forecast for {item_code}',
        xaxis_title='Date',
        yaxis_title='Quantity',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_inventory_status(recommendations_df):
    """Plot inventory status overview"""
    # Items needing reorder
    reorder_counts = recommendations_df['needs_reorder'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Needs Reorder', 'Stock OK'],
        values=[reorder_counts.get(True, 0), reorder_counts.get(False, 0)],
        hole=0.4,
        marker=dict(colors=['#dc3545', '#28a745'])
    )])
    
    fig.update_layout(
        title='Inventory Status Overview',
        height=400
    )
    
    return fig


def plot_top_items(recommendations_df, metric='days_until_stockout', top_n=10):
    """Plot top items by specified metric"""
    if metric == 'days_until_stockout':
        top_items = recommendations_df.nsmallest(top_n, metric)
        title = f'Top {top_n} Items by Urgency (Days Until Stockout)'
        x_label = 'Days Until Stockout'
    else:
        top_items = recommendations_df.nlargest(top_n, metric)
        title = f'Top {top_n} Items by {metric.replace("_", " ").title()}'
        x_label = metric.replace('_', ' ').title()
    
    fig = go.Figure(data=[go.Bar(
        x=top_items[metric],
        y=top_items['item_description'],
        orientation='h',
        marker=dict(color='#1f77b4')
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='Item',
        height=400,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 AI-Driven Inventory Management System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI-powered inventory management system for plant maintenance. 
    This system uses machine learning to forecast demand and optimize inventory levels.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Inventory Data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your inventory data file"
        )
        
        # Load sample data option
        use_sample = st.checkbox("Use Sample Data", value=False)
        
        # Forecasting parameters
        st.subheader("🔮 Forecasting Settings")
        forecast_models = st.multiselect(
            "Select Forecasting Models",
            ['arima', 'prophet', 'gradient_boosting'],
            default=['arima', 'gradient_boosting'],
            help="Choose which models to use for forecasting"
        )
        
        service_level = st.slider(
            "Service Level (%)",
            min_value=85,
            max_value=99,
            value=int(DEFAULT_SERVICE_LEVEL * 100),
            help="Target service level for inventory management"
        ) / 100
        
        # Email configuration
        st.subheader("📧 Email Settings")
        with st.expander("Configure Email Notifications"):
            sender_email = st.text_input("Sender Email (Outlook)", 
                                        help="Your Outlook email address")
            sender_password = st.text_input("Email Password", 
                                           type="password",
                                           help="Your Outlook password or app password")
            recipient_email = st.text_input("Recipient Email",
                                           help="Email address to receive alerts")
            
            if st.button("Test Email Connection"):
                if sender_email and sender_password:
                    with st.spinner("Testing connection..."):
                        email_service = EmailService(sender_email, sender_password)
                        result = email_service.test_connection()
                        if result['success']:
                            st.success(result['message'])
                            st.session_state.email_configured = True
                        else:
                            st.error(result['message'])
                else:
                    st.warning("Please enter email credentials")
    
    # Main content
    tabs = st.tabs(["📊 Data Processing", "🔮 Demand Forecasting", 
                    "📦 Inventory Analysis", "📧 Email Notifications", 
                    "📈 Dashboard"])
    
    # Tab 1: Data Processing
    with tabs[0]:
        st.header("Data Processing & Preprocessing")
        
        # Load data
        df = None
        if use_sample:
            try:
                df = pd.read_csv('sample_inventory_data.csv')
                st.success("✅ Sample data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        elif uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.success("✅ Data loaded successfully!")
        
        if df is not None:
            # Display overview
            display_data_overview(df)
            
            # Process data
            if st.button("🔄 Process Data", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        processed_data, item_summary, validation_report = preprocess_inventory_data(df)
                        
                        st.session_state.processed_data = processed_data
                        st.session_state.item_summary = item_summary
                        
                        st.success("✅ Data processing complete!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Processed Data Summary")
                            st.metric("Processed Records", f"{len(processed_data):,}")
                            st.metric("Unique Items", f"{len(item_summary):,}")
                        
                        with col2:
                            st.subheader("Validation Report")
                            if validation_report['is_valid']:
                                st.success("✅ Data validation passed")
                            else:
                                st.error("❌ Data validation failed")
                                for error in validation_report.get('errors', []):
                                    st.error(error)
                        
                        # Display item summary
                        with st.expander("📋 View Item Summary"):
                            st.dataframe(item_summary, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        else:
            st.info("👆 Please upload data or select 'Use Sample Data' to begin")
    
    # Tab 2: Demand Forecasting
    with tabs[1]:
        st.header("Demand Forecasting")
        
        if st.session_state.item_summary is not None:
            # Select item for forecasting
            items = st.session_state.item_summary['item_code'].tolist()
            selected_item = st.selectbox("Select Item for Forecasting", items)
            
            if selected_item:
                item_info = st.session_state.item_summary[
                    st.session_state.item_summary['item_code'] == selected_item
                ].iloc[0]
                
                # Display item info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Item Code", selected_item)
                with col2:
                    st.metric("Current Stock", f"{item_info['stock_onhand']:.2f}")
                with col3:
                    st.metric("Unit Cost", format_currency(item_info['unit_cost']))
                
                st.write(f"**Description:** {item_info['item_description']}")
                st.write(f"**Equipment:** {item_info['equipment_name']}")
                
                # Forecast button
                if st.button("🔮 Generate Forecast", type="primary"):
                    with st.spinner(f"Forecasting demand for {selected_item}..."):
                        try:
                            # Prepare time series data
                            preprocessor = InventoryDataPreprocessor(st.session_state.processed_data)
                            ts_data = preprocessor.prepare_time_series(
                                st.session_state.processed_data,
                                selected_item
                            )
                            
                            if len(ts_data) < MIN_HISTORICAL_DAYS:
                                st.warning(f"⚠️ Insufficient historical data for {selected_item}. "
                                         f"Minimum {MIN_HISTORICAL_DAYS} days required, found {len(ts_data)} days.")
                            else:
                                # Run forecasting
                                results = forecast_demand(
                                    ts_data,
                                    selected_item,
                                    item_info['item_description'],
                                    models=forecast_models
                                )
                                
                                # Store results
                                st.session_state.forecast_results[selected_item] = results
                                
                                st.success("✅ Forecast generated successfully!")
                                
                                # Display results
                                for model_name, model_result in results.items():
                                    if 'error' not in model_result:
                                        st.subheader(f"📊 {model_name.upper()} Model Results")
                                        
                                        if 'forecast' in model_result:
                                            forecast_df = model_result['forecast']
                                            
                                            # Metrics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                if 'rmse' in model_result:
                                                    st.metric("RMSE", f"{model_result['rmse']:.2f}")
                                            with col2:
                                                if 'mae' in model_result:
                                                    st.metric("MAE", f"{model_result['mae']:.2f}")
                                            with col3:
                                                total_forecast = forecast_df['forecasted_quantity'].sum()
                                                st.metric("30-Day Forecast", f"{total_forecast:.2f}")
                                            
                                            # Plot
                                            fig = plot_demand_forecast(selected_item, ts_data, forecast_df)
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Forecast table
                                            with st.expander("📋 View Forecast Data"):
                                                st.dataframe(forecast_df, use_container_width=True)
                                    else:
                                        st.error(f"❌ {model_name}: {model_result['error']}")
                        
                        except Exception as e:
                            st.error(f"Error generating forecast: {str(e)}")
                
                # Forecast all items
                st.divider()
                if st.button("🔮 Forecast All Items", type="secondary"):
                    with st.spinner("Forecasting demand for all items..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_items = len(items)
                        for idx, item_code in enumerate(items):
                            status_text.text(f"Processing {item_code} ({idx+1}/{total_items})...")
                            
                            try:
                                # Prepare time series
                                preprocessor = InventoryDataPreprocessor(st.session_state.processed_data)
                                ts_data = preprocessor.prepare_time_series(
                                    st.session_state.processed_data,
                                    item_code
                                )
                                
                                if len(ts_data) >= MIN_HISTORICAL_DAYS:
                                    item_desc = st.session_state.item_summary[
                                        st.session_state.item_summary['item_code'] == item_code
                                    ]['item_description'].iloc[0]
                                    
                                    results = forecast_demand(
                                        ts_data,
                                        item_code,
                                        item_desc,
                                        models=forecast_models
                                    )
                                    
                                    st.session_state.forecast_results[item_code] = results
                            
                            except Exception as e:
                                st.warning(f"⚠️ Error forecasting {item_code}: {str(e)}")
                            
                            progress_bar.progress((idx + 1) / total_items)
                        
                        status_text.text("Forecasting complete!")
                        st.success(f"✅ Forecasted {len(st.session_state.forecast_results)} items successfully!")
        else:
            st.info("👆 Please process data first in the 'Data Processing' tab")
    
    # Tab 3: Inventory Analysis
    with tabs[2]:
        st.header("Inventory Analysis & Recommendations")
        
        if st.session_state.forecast_results:
            if st.button("📊 Generate Recommendations", type="primary"):
                with st.spinner("Generating inventory recommendations..."):
                    try:
                        recommendations = generate_reorder_recommendations(
                            st.session_state.item_summary,
                            st.session_state.forecast_results,
                            service_level=service_level
                        )
                        
                        st.session_state.recommendations = recommendations
                        
                        st.success("✅ Recommendations generated successfully!")
                        
                        # Calculate metrics
                        metrics = calculate_inventory_value(recommendations)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Items Needing Reorder", 
                                     f"{metrics['items_needing_reorder']}/{metrics['total_items']}")
                        with col2:
                            st.metric("Current Inventory Value", 
                                     format_currency(metrics['total_current_inventory_value']))
                        with col3:
                            st.metric("Reorder Value", 
                                     format_currency(metrics['total_reorder_value']))
                        with col4:
                            st.metric("Avg Turnover Ratio", 
                                     f"{metrics['average_turnover_ratio']:.2f}")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = plot_inventory_status(recommendations)
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            fig2 = plot_top_items(recommendations, 'days_until_stockout')
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Recommendations table
                        st.subheader("📋 Detailed Recommendations")
                        
                        # Filter options
                        show_all = st.checkbox("Show all items", value=False)
                        
                        if show_all:
                            display_df = recommendations
                        else:
                            display_df = recommendations[recommendations['needs_reorder'] == True]
                        
                        # Display table
                        st.dataframe(
                            display_df[[
                                'item_code', 'item_description', 'equipment_name',
                                'current_stock', 'reorder_point', 'recommended_order_quantity',
                                'days_until_stockout', 'forecasted_30day_demand', 'needs_reorder'
                            ]],
                            use_container_width=True
                        )
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = recommendations.to_csv(index=False)
                            st.download_button(
                                "📥 Download as CSV",
                                csv,
                                "inventory_recommendations.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        
                        with col2:
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                recommendations.to_excel(writer, index=False, sheet_name='Recommendations')
                            
                            st.download_button(
                                "📥 Download as Excel",
                                buffer.getvalue(),
                                "inventory_recommendations.xlsx",
                                "application/vnd.ms-excel",
                                key='download-excel'
                            )
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
        else:
            st.info("👆 Please generate forecasts first in the 'Demand Forecasting' tab")
    
    # Tab 4: Email Notifications
    with tabs[3]:
        st.header("Email Notifications")
        
        if st.session_state.recommendations is not None:
            items_to_reorder = st.session_state.recommendations[
                st.session_state.recommendations['needs_reorder'] == True
            ]
            
            if len(items_to_reorder) > 0:
                st.write(f"**{len(items_to_reorder)} items require reordering**")
                
                # Email settings check
                if not st.session_state.email_configured:
                    st.warning("⚠️ Please configure email settings in the sidebar first")
                else:
                    # Email options
                    batch_mode = st.radio(
                        "Email Mode",
                        ["Send one email with all items", "Send individual emails for each item"],
                        help="Choose how to send reorder notifications"
                    )
                    
                    batch_mode_bool = (batch_mode == "Send one email with all items")
                    
                    # Preview email
                    with st.expander("📧 Preview Email"):
                        if batch_mode_bool:
                            st.write("**Subject:** Inventory Reorder Alert: Multiple Items Require Attention")
                            st.write("**Body Preview:**")
                            st.text(f"""
Dear Inventory Manager,

This is an automated alert for {len(items_to_reorder)} items that require reordering:

1. Item: {items_to_reorder.iloc[0]['item_description']}
   Item Code: {items_to_reorder.iloc[0]['item_code']}
   Current Stock: {items_to_reorder.iloc[0]['current_stock']:.2f}
   Recommended Order: {items_to_reorder.iloc[0]['recommended_order_quantity']:.2f}
   
... (and {len(items_to_reorder)-1} more items)

A detailed CSV report is attached.
                            """)
                        else:
                            st.write("**Subject:** Inventory Reorder Alert: [Item Description]")
                            st.write("**Body Preview:**")
                            st.text(f"""
Dear Inventory Manager,

This is an automated alert for the following item:

Item Code: {items_to_reorder.iloc[0]['item_code']}
Item Description: {items_to_reorder.iloc[0]['item_description']}
Current Stock: {items_to_reorder.iloc[0]['current_stock']:.2f}
Recommended Order: {items_to_reorder.iloc[0]['recommended_order_quantity']:.2f}
                            """)
                    
                    # Send emails
                    if st.button("📧 Send Email Notifications", type="primary"):
                        with st.spinner("Sending emails..."):
                            try:
                                # Get email credentials from sidebar
                                sender_email = st.session_state.get('sender_email', '')
                                sender_password = st.session_state.get('sender_password', '')
                                recipient_email = st.session_state.get('recipient_email', '')
                                
                                # Note: In production, you'd retrieve these securely
                                # For demo, we'll show a placeholder
                                st.warning("⚠️ Email sending is disabled in demo mode. "
                                         "In production, emails would be sent using configured credentials.")
                                
                                # Simulate sending
                                st.success(f"✅ Would send {len(items_to_reorder)} email notifications")
                                
                                # Show what would be sent
                                with st.expander("📋 Items that would receive notifications"):
                                    st.dataframe(
                                        items_to_reorder[[
                                            'item_code', 'item_description', 
                                            'current_stock', 'recommended_order_quantity'
                                        ]],
                                        use_container_width=True
                                    )
                            
                            except Exception as e:
                                st.error(f"Error sending emails: {str(e)}")
            else:
                st.success("✅ No items require reordering at this time!")
        else:
            st.info("👆 Please generate recommendations first in the 'Inventory Analysis' tab")
    
    # Tab 5: Dashboard
    with tabs[4]:
        st.header("Inventory Management Dashboard")
        
        if st.session_state.recommendations is not None:
            recommendations = st.session_state.recommendations
            
            # Key metrics
            st.subheader("📊 Key Performance Indicators")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_items = len(recommendations)
                st.metric("Total Items", f"{total_items:,}")
            
            with col2:
                items_needing_reorder = recommendations['needs_reorder'].sum()
                st.metric("Items Needing Reorder", f"{items_needing_reorder:,}")
            
            with col3:
                avg_stock = recommendations['current_stock'].mean()
                st.metric("Avg Stock Level", f"{avg_stock:.2f}")
            
            with col4:
                total_value = (recommendations['current_stock'] * recommendations['unit_cost']).sum()
                st.metric("Total Inventory Value", format_currency(total_value))
            
            with col5:
                avg_turnover = recommendations['inventory_turnover_ratio'].mean()
                st.metric("Avg Turnover Ratio", f"{avg_turnover:.2f}")
            
            # Charts
            st.subheader("📈 Visual Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Inventory status
                fig1 = plot_inventory_status(recommendations)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Top items by urgency
                fig3 = plot_top_items(recommendations, 'days_until_stockout', top_n=10)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Stock distribution
                fig2 = px.histogram(
                    recommendations,
                    x='current_stock',
                    nbins=20,
                    title='Stock Level Distribution',
                    labels={'current_stock': 'Current Stock', 'count': 'Number of Items'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Turnover ratio distribution
                fig4 = px.box(
                    recommendations,
                    y='inventory_turnover_ratio',
                    title='Inventory Turnover Ratio Distribution',
                    labels={'inventory_turnover_ratio': 'Turnover Ratio'}
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            # Critical items
            st.subheader("⚠️ Critical Items (Urgent Attention Required)")
            critical_items = recommendations[recommendations['days_until_stockout'] < 7]
            
            if len(critical_items) > 0:
                st.error(f"🚨 {len(critical_items)} items will stock out within 7 days!")
                st.dataframe(
                    critical_items[[
                        'item_code', 'item_description', 'current_stock',
                        'days_until_stockout', 'recommended_order_quantity'
                    ]].sort_values('days_until_stockout'),
                    use_container_width=True
                )
            else:
                st.success("✅ No critical items at this time")
        else:
            st.info("👆 Please complete the analysis workflow to view the dashboard")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>AI-Driven Inventory Management System | Powered by Machine Learning</p>
        <p>© 2025 Plant Maintenance Solutions</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
