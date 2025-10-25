import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from smart_portfolio_analyzer import (
    Portfolio, 
    StockAsset, 
    BondAsset,
    RiskSimulator,
    DataManager,
    Analytics
)

# Page config
st.set_page_config(
    page_title="Smart Portfolio Analyzer & Risk Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size:24px; color: #1f77b4; font-weight: bold;}
    .section-header {font-size:20px; color: #2ca02c; margin-top: 20px;}
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio("My Portfolio")
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

# Sidebar
st.sidebar.title("Portfolio Manager")

# Main app
st.title("📊 Smart Portfolio Analyzer & Risk Simulator")
st.markdown("---")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Portfolio Overview", 
    "📈 Performance", 
    "📊 Risk Analysis", 
    "➕ Add Assets"
])

# Tab 1: Portfolio Overview
with tab1:
    st.header("📋 Portfolio Overview")
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${st.session_state.portfolio.get_total_value():,.2f}")
    with col2:
        st.metric("Number of Assets", len(st.session_state.portfolio.assets))
    with col3:
        st.metric("Total P&L", 
                 f"${sum(asset.profit_loss for asset in st.session_state.portfolio.assets.values()):,.2f}",
                 f"{sum(asset.profit_loss for asset in st.session_state.portfolio.assets.values()) / sum(asset.cost_basis for asset in st.session_state.portfolio.assets.values()) * 100 if sum(asset.cost_basis for asset in st.session_state.portfolio.assets.values()) > 0 else 0:.2f}%"
                 )
    
    # Asset allocation
    st.subheader("Asset Allocation")
    if st.session_state.portfolio.assets:
        # Pie chart for asset allocation
        allocation = st.session_state.portfolio.get_asset_allocation()
        fig = px.pie(
            names=list(allocation.keys()),
            values=list(allocation.values()),
            title="Portfolio Allocation by Asset"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Asset table
        st.subheader("Asset Details")
        asset_data = []
        for asset_id, asset in st.session_state.portfolio.assets.items():
            asset_data.append({
                "Symbol": asset.symbol,
                "Name": asset.name,
                "Type": asset.get_asset_type(),
                "Quantity": asset.quantity,
                "Avg. Cost": f"${asset.purchase_price:,.2f}",
                "Current Price": f"${asset.current_price:,.2f}",
                "Market Value": f"${asset.market_value:,.2f}",
                "P&L": f"${asset.profit_loss:,.2f}",
                "P&L %": f"{asset.profit_loss_pct:.2f}%"
            })
        
        df_assets = pd.DataFrame(asset_data)
        st.dataframe(df_assets, use_container_width=True)
    else:
        st.info("No assets in portfolio. Add assets using the 'Add Assets' tab.")

# Tab 2: Performance
with tab2:
    st.header("📈 Performance Analysis")
    
    if not st.session_state.portfolio.assets:
        st.warning("Add assets to your portfolio to see performance metrics.")
    else:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                     value=datetime.now() - timedelta(days=365),
                                     max_value=datetime.now() - timedelta(days=1))
        with col2:
            end_date = st.date_input("End Date", 
                                   value=datetime.now(),
                                   min_value=start_date + timedelta(days=1),
                                   max_value=datetime.now())
        
        # Get historical data
        symbols = [asset.symbol for asset in st.session_state.portfolio.assets.values()]
        historical_data = st.session_state.data_manager.get_historical_prices(
            symbols=symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if historical_data:
            # Portfolio value over time
            st.subheader("Portfolio Value Over Time")
            portfolio_values = pd.DataFrame()
            for symbol, data in historical_data.items():
                if not data.empty:
                    portfolio_values[symbol] = data['close'] * \
                        next((a.quantity for a in st.session_state.portfolio.assets.values() 
                             if a.symbol == symbol), 0)
            
            if not portfolio_values.empty:
                portfolio_values['Total'] = portfolio_values.sum(axis=1)
                fig = px.line(portfolio_values, y='Total', 
                             title="Portfolio Value Over Time",
                             labels={'value': 'Value ($)', 'index': 'Date'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                returns = portfolio_values['Total'].pct_change().dropna()
                metrics = Analytics.calculate_performance_metrics(
                    returns.values,
                    risk_free_rate=st.session_state.data_manager.get_risk_free_rate()
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{(portfolio_values['Total'].iloc[-1] / portfolio_values['Total'].iloc[0] - 1) * 100:.2f}%")
                    st.metric("Annualized Return", f"{metrics.mean_return:.2%}")
                with col2:
                    st.metric("Volatility", f"{metrics.volatility:.2%}")
                    st.metric("Max Drawdown", f"{-metrics.max_drawdown:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
                    st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
                with col4:
                    st.metric("95% VaR (1-day)", f"{-metrics.var_95*portfolio_values['Total'].iloc[-1]:.2f}")
                    st.metric("95% CVaR (1-day)", f"{-metrics.cvar_95*portfolio_values['Total'].iloc[-1]:.2f}")

# Tab 3: Risk Analysis
with tab3:
    st.header("📊 Risk Analysis")
    
    if not st.session_state.portfolio.assets:
        st.warning("Add assets to your portfolio to perform risk analysis.")
    else:
        # Monte Carlo Simulation
        st.subheader("Monte Carlo Simulation")
        col1, col2 = st.columns(2)
        with col1:
            num_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
        with col2:
            time_horizon = st.slider("Time Horizon (days)", 30, 252*5, 252, 30)
        
        if st.button("Run Simulation"):
            with st.spinner("Running Monte Carlo simulation..."):
                simulator = RiskSimulator(st.session_state.portfolio)
                result = simulator.run_monte_carlo_simulation(
                    num_simulations=num_simulations,
                    time_horizon=time_horizon
                )
                
                # Store results in session state
                st.session_state.simulation_results = result
                
            # Display results
            st.success("Simulation completed successfully!")
            
            # Plot simulation results
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=result.simulated_returns * 100,
                nbinsx=50,
                name='Returns Distribution',
                marker_color='#1f77b4',
                opacity=0.7
            ))
            
            # Add VaR lines
            for cl, var in result.var.items():
                var_pct = var * 100
                fig.add_vline(
                    x=var_pct,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{int(cl*100)}% VaR: {var_pct:.2f}%",
                    annotation_position="top right"
                )
            
            fig.update_layout(
                title="Portfolio Returns Distribution",
                xaxis_title="Portfolio Return (%)",
                yaxis_title="Frequency",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            st.subheader("Risk Metrics")
            metrics = simulator.estimate_risk_metrics(time_horizon=time_horizon)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Return", f"{metrics['expected_return']:.2%}")
                st.metric("Volatility", f"{metrics['volatility']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown", f"{-metrics['max_drawdown']:.2%}")
            with col3:
                st.metric("95% VaR", f"{-metrics['var'][0.95]*100:.2f}%")
                st.metric("95% CVaR", f"{-metrics['cvar'][0.95]*100:.2f}%")
            
            # Stress Testing
            st.subheader("Stress Testing")
            stress_scenarios = simulator.calculate_stress_scenarios()
            
            for scenario, results in stress_scenarios.items():
                st.metric(
                    f"Scenario: {scenario.replace('_', ' ').title()}",
                    f"-${-results['dollar_impact']:,.2f}",
                    f"{results['percent_impact']:.2f}%"
                )

# Tab 4: Add Assets
with tab4:
    st.header("➕ Add Assets to Portfolio")
    
    asset_type = st.radio("Asset Type", ["Stock", "Bond"])
    
    with st.form("add_asset_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol (e.g., AAPL)", "").upper()
            name = st.text_input("Company/Asset Name", "")
            quantity = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
            purchase_price = st.number_input("Purchase Price ($)", min_value=0.0, value=0.0)
            current_price = st.number_input("Current Price ($)", min_value=0.0, value=0.0)
            purchase_date = st.date_input("Purchase Date", value=datetime.now())
        
        with col2:
            if asset_type == "Stock":
                sector = st.selectbox(
                    "Sector",
                    ["", "Technology", "Healthcare", "Financial", "Consumer Cyclical", 
                     "Industrial", "Communication Services", "Energy", "Utilities", 
                     "Real Estate", "Materials", "Consumer Defensive"]
                )
                dividend_yield = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0) / 100
                beta = st.number_input("Beta", min_value=0.0, value=1.0)
            else:  # Bond
                coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, value=3.0) / 100
                face_value = st.number_input("Face Value ($)", min_value=0.0, value=1000.0)
                maturity_date = st.date_input("Maturity Date", 
                                             value=datetime.now() + timedelta(days=365*10))
                credit_rating = st.selectbox(
                    "Credit Rating",
                    ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
                     "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]
                )
                payment_frequency = st.selectbox(
                    "Payment Frequency",
                    [1, 2, 4, 12],
                    index=1  # Default to semi-annual
                )
        
        if st.form_submit_button("Add Asset"):
            try:
                if asset_type == "Stock":
                    asset = StockAsset(
                        asset_id=f"{symbol}_{int(datetime.now().timestamp())}",
                        name=name,
                        symbol=symbol,
                        current_price=current_price,
                        purchase_price=purchase_price,
                        quantity=quantity,
                        purchase_date=purchase_date,
                        sector=sector,
                        dividend_yield=dividend_yield,
                        beta=beta
                    )
                else:  # Bond
                    asset = BondAsset(
                        asset_id=f"BOND_{symbol}_{int(datetime.now().timestamp())}",
                        name=name,
                        symbol=symbol,
                        current_price=current_price,
                        purchase_price=purchase_price,
                        quantity=quantity,
                        purchase_date=purchase_date,
                        coupon_rate=coupon_rate,
                        face_value=face_value,
                        maturity_date=maturity_date,
                        credit_rating=credit_rating,
                        payment_frequency=payment_frequency
                    )
                
                st.session_state.portfolio.add_asset(asset)
                st.success(f"Successfully added {quantity} shares of {symbol} to your portfolio!")
                
            except Exception as e:
                st.error(f"Error adding asset: {str(e)}")

# Run the app
if __name__ == "__main__":
    st.write("Smart Portfolio Analyzer is running...")
