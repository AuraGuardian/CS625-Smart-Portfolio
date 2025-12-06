import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from smart_portfolio_analyzer import (
    Portfolio, 
    StockAsset, 
    BondAsset,
    RiskSimulator,
    DataManager,
    PortfolioAnalyzer
)

# Initialize DataManager with environment variables
try:
    # Try to get API key from Streamlit secrets first (for production)
    if 'POLYGON_API_KEY' in st.secrets:
        DATA_MANAGER = DataManager(api_key=st.secrets['POLYGON_API_KEY'])
    # Fallback to .env file for local development
    else:
        from dotenv import load_dotenv
        import os
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment variables
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
            
        DATA_MANAGER = DataManager(api_key=polygon_api_key)
        
except Exception as e:
    st.error(f"Error initializing DataManager: {str(e)}")
    st.error("Please ensure you have a valid Polygon.io API key in Streamlit secrets or .env file")
    st.stop()

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
    st.session_state.data_manager = DATA_MANAGER

# Sidebar
st.sidebar.title("Portfolio Manager")

# Main app
st.title("📊 Smart Portfolio Analyzer & Risk Simulator")
st.markdown("---")

# Tab layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Portfolio Overview", 
    "📈 Performance", 
    "📊 Risk Analysis",
    "➕ Add Assets",
    "📈 Efficient Frontier",
    "🔮 Price Forecast"  # New tab for Prophet forecasting
])

# Tab 1: Portfolio Overview
with tab1:
    st.header("📋 Portfolio Overview")
    
    # Add button to load sample data if portfolio is empty
    if not st.session_state.portfolio.assets:
        if st.button("📊 Load Sample Portfolio"):
            try:
                try:
                    import os
                    import json
                    from pathlib import Path
                    from smart_portfolio_analyzer.portfolio import Portfolio
                    from smart_portfolio_analyzer.assets import StockAsset, BondAsset
                    
                    # Get the absolute path to the sample data file
                    base_dir = Path(__file__).parent.absolute()
                    sample_file = base_dir / "sample_data" / "sample_portfolio.json"
                    
                    # Debug: Print the file path being used
                    print(f"Looking for sample file at: {sample_file}")
                    
                    if not sample_file.exists():
                        st.error(f"Sample data file not found at: {sample_file}")
                        # Continue with the rest of the code to show additional error details
                    
                    # Load the sample data
                    with open(sample_file, 'r') as f:
                        sample_data = json.load(f)
                    
                    # Debug: Print the loaded data
                    print(f"Loaded sample data: {json.dumps(sample_data, indent=2)[:500]}...")
                    
                    # Create a new portfolio with the sample data
                    portfolio = Portfolio(
                        name=sample_data.get('name', 'Sample Portfolio'),
                        description=sample_data.get('description', ''),
                        risk_free_rate=float(sample_data.get('risk_free_rate', 0.02))
                    )
                    
                    # Add each asset to the portfolio
                    assets_added = 0
                    for i, asset_data in enumerate(sample_data.get('assets', [])):
                        try:
                            asset_type = asset_data.get('asset_type', '').lower()
                            ticker = asset_data.get('ticker', 'unknown')
                            
                            # Add required ID field if missing
                            if 'id' not in asset_data:
                                asset_data['id'] = f"{ticker}_{i}"
                            
                            try:
                                if asset_type in ['stock', 'etf']:
                                    # For both stocks and ETFs, we can use StockAsset
                                    asset_data.setdefault('exchange', 'NYSE')
                                    asset = StockAsset.from_dict(asset_data)
                                    print(f"Created {asset_type.upper()}: {ticker}")
                                elif asset_type == 'bond':
                                    # Ensure required fields are present with defaults
                                    asset_data.setdefault('issuer', 'Unknown')
                                    asset_data.setdefault('maturity_date', '2030-01-01')
                                    asset_data.setdefault('coupon_rate', 0.0)
                                    asset = BondAsset.from_dict(asset_data)
                                    print(f"Created bond: {ticker}")
                                else:
                                    print(f"Skipping unknown asset type: {asset_type}")
                                    continue
                            except Exception as e:
                                print(f"Error creating asset {ticker}: {str(e)}")
                                continue
                                
                            # Add the asset with its weight if available
                            weight = sample_data.get('weights', {}).get(ticker)
                            portfolio.add_asset(asset, weight)
                            assets_added += 1
                            print(f"Added asset: {ticker} (type: {asset_type}, weight: {weight})")
                            
                        except Exception as e:
                            print(f"Error adding asset {ticker}: {str(e)}")
                            st.error(f"Error adding asset {ticker}: {str(e)}")
                    
                    if assets_added > 0:
                        st.session_state.portfolio = portfolio
                        st.success(f"Successfully loaded {assets_added} assets into the portfolio!")
                        st.rerun()
                    else:
                        st.error("No assets were added to the portfolio. Please check the sample data file.")
                        
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Error loading sample portfolio: {error_details}")
                    st.error(f"Error loading sample portfolio: {str(e)}")
                    st.text_area("Error details", error_details, height=200)
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${st.session_state.portfolio.get_total_value():,.2f}")
    with col2:
        st.metric("Number of Assets", len(st.session_state.portfolio.assets))
    with col3:
        if st.session_state.portfolio.assets:
            total_pl = sum(asset.profit_loss() for asset in st.session_state.portfolio.assets)
            total_cost_basis = sum(asset.cost_basis() for asset in st.session_state.portfolio.assets)
            pl_percent = (total_pl / total_cost_basis * 100) if total_cost_basis > 0 else 0
            st.metric(
                "Total P&L", 
                f"${total_pl:,.2f}",
                f"{pl_percent:.2f}%"
            )
        else:
            st.metric("Total P&L", "$0.00", "0.00%")
    
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
        for asset in st.session_state.portfolio.assets:
            asset_data.append({
                "Symbol": asset.ticker,
                "Name": asset.name,
                "Type": asset.asset_type,
                "Quantity": asset._quantity,
                "Purchase Price": f"${asset._purchase_price:,.2f}",
                "Current Price": f"${asset._current_price:,.2f}" if asset._current_price is not None else "N/A",
                "Market Value": f"${asset.current_value():,.2f}",
                "P&L": f"${asset.profit_loss():,.2f}",
                "P&L %": f"{asset.profit_loss_percent() * 100:.2f}%"
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
        symbols = [getattr(asset, 'symbol', getattr(asset, 'ticker', '')) 
                 for asset in st.session_state.portfolio.assets]
        symbols = [s for s in symbols if s]  # Remove empty symbols
        
        if not symbols:
            st.warning("No valid assets with symbols found in the portfolio.")
        else:
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
                    if not data.empty and 'close' in data.columns:
                        # Find the asset by symbol or ticker
                        asset_quantity = 0
                        for asset in st.session_state.portfolio.assets:
                            asset_symbol = getattr(asset, 'symbol', getattr(asset, 'ticker', ''))
                            if asset_symbol == symbol:
                                asset_quantity = getattr(asset, 'quantity', 0)
                                break
                        
                        if asset_quantity > 0:
                            portfolio_values[symbol] = data['close'] * asset_quantity
            
            if not portfolio_values.empty:
                portfolio_values['Total'] = portfolio_values.sum(axis=1)
                fig = px.line(portfolio_values, y='Total', 
                             title="Portfolio Value Over Time",
                             labels={'value': 'Value ($)', 'index': 'Date'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate performance metrics
                st.subheader("Performance Metrics")
                
                # Ensure we have enough data points
                if len(portfolio_values) > 1 and 'Total' in portfolio_values.columns:
                    try:
                        # Get risk-free rate with fallback
                        try:
                            risk_free_rate = st.session_state.data_manager.get_risk_free_rate()
                        except Exception as e:
                            st.warning(f"Could not fetch risk-free rate, using 0%: {str(e)}")
                            risk_free_rate = 0.0
                        
                        # Calculate returns and metrics
                        returns = portfolio_values['Total'].pct_change().dropna()
                        if not returns.empty:
                            metrics = PortfolioAnalyzer.calculate_performance_metrics(
                                returns.values,
                                risk_free_rate=risk_free_rate
                            )
                            
                            # Calculate total return safely
                            total_return = 0.0
                            if portfolio_values['Total'].iloc[0] != 0:
                                total_return = (portfolio_values['Total'].iloc[-1] / portfolio_values['Total'].iloc[0] - 1) * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Return", f"{total_return:.2f}%")
                                st.metric("Annualized Return", f"{metrics.mean_return:.2%}")
                            with col2:
                                st.metric("Volatility", f"{metrics.volatility:.2%}")
                                st.metric("Max Drawdown", f"{-metrics.max_drawdown:.2%}")
                            with col3:
                                st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
                                st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
                            with col4:
                                var_95 = -metrics.var_95 * portfolio_values['Total'].iloc[-1] if hasattr(metrics, 'var_95') and metrics.var_95 is not None else 0
                                cvar_95 = -metrics.cvar_95 * portfolio_values['Total'].iloc[-1] if hasattr(metrics, 'cvar_95') and metrics.cvar_95 is not None else 0
                                st.metric("95% VaR (1-day)", f"${max(0, var_95):.2f}")
                                st.metric("95% CVaR (1-day)", f"${max(0, cvar_95):.2f}")
                    except Exception as e:
                        st.error(f"Error calculating performance metrics: {str(e)}")

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

# Tab 5: Efficient Frontier
with tab5:
    st.header("📈 Efficient Frontier")
    
    if not st.session_state.portfolio.assets:
        st.warning("Please add assets to your portfolio to view the efficient frontier.")
    else:
        # Get historical returns for all assets
        symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
        try:
            # Get 1 year of historical data
            with st.spinner("Calculating efficient frontier..."):
                # Get historical prices for all assets in the portfolio
                symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
                
                # Get 1 year of historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                # Fetch historical data
                historical_data = st.session_state.data_manager.get_historical_prices(
                    symbols=symbols,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if not historical_data:
                    st.error("No historical data returned. Please check your data source.")
                    st.stop()
                
                # Get the first ticker's data
                first_ticker = next(iter(historical_data))
                first_value = historical_data[first_ticker]
                
                # Handle case where values are DataFrames
                if isinstance(first_value, pd.DataFrame):
                    # Find the close column (case insensitive)
                    close_col = next((col for col in first_value.columns if str(col).lower() == 'close'), None)
                    
                    if close_col is not None:
                        # Extract close prices from each DataFrame
                        close_prices = {
                            ticker: df[close_col].values 
                            for ticker, df in historical_data.items()
                            if isinstance(df, pd.DataFrame) and close_col in df.columns
                        }
                        
                        if close_prices:
                            prices_df = pd.DataFrame(close_prices)
                        else:
                            st.error("No valid price data found in the DataFrames")
                            st.stop()
                    else:
                        st.error(f"No 'close' column found in the DataFrames. Available columns: {first_value.columns.tolist()}")
                        st.stop()
                else:
                    st.error(f"Unsupported data type: {type(first_value).__name__}")
                    st.stop()
                
                # Ensure we have numeric data
                prices_df = prices_df.apply(pd.to_numeric, errors='coerce')
                
                # Drop rows with any missing values
                prices_df = prices_df.dropna(how='any')
                
                if len(prices_df) < 2:
                    st.error("Not enough data points to calculate returns. Try with a longer time period.")
                    st.stop()
                
                # Calculate daily returns
                returns = prices_df.pct_change().dropna()
                
                if len(returns) < 2:
                    st.error("Not enough data to calculate returns. Try with a longer time period.")
                    st.stop()
                
                # Calculate expected returns (annualized)
                expected_returns = returns.mean() * 252
                
                # Calculate covariance matrix (annualized)
                cov_matrix = returns.cov() * 252
                
                # Ensure we have valid data
                if expected_returns.isna().any() or cov_matrix.isna().any().any():
                    st.error("Error: Invalid data in returns or covariance matrix calculation.")
                    st.write("Expected returns:", expected_returns)
                    st.write("Covariance matrix:", cov_matrix)
                    st.stop()
                
                # Number of portfolios to simulate
                num_portfolios = 10000
            
            # Store results
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            # Generate random portfolios
            for i in range(num_portfolios):
                # Random weights
                weights = np.random.random(len(symbols))
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                # Portfolio return and volatility
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Store results
                results[0, i] = portfolio_volatility
                results[1, i] = portfolio_return
                results[2, i] = results[1, i] / (results[0, i] + 1e-10)  # Sharpe ratio (with epsilon to avoid division by zero)
                
                # Get current portfolio weights
                current_weights = np.array([
                    asset.current_value() / st.session_state.portfolio.get_total_value() 
                    for asset in st.session_state.portfolio.assets
                ])
                
            # Calculate current portfolio metrics
            current_weights = np.array([
                asset.current_value() / st.session_state.portfolio.get_total_value() 
                for asset in st.session_state.portfolio.assets
            ])
            
            current_return = np.sum(current_weights * expected_returns)
            current_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
            current_sharpe = current_return / (current_volatility + 1e-10)
            
            # Find optimal portfolio (max Sharpe ratio)
            max_sharpe_idx = np.argmax(results[2])
            optimal_weights = weights_record[max_sharpe_idx]
            optimal_return = results[1, max_sharpe_idx]
            optimal_volatility = results[0, max_sharpe_idx]
            optimal_sharpe = results[2, max_sharpe_idx]
            
            # Create efficient frontier plot
            fig = go.Figure()
            
            # Plot random portfolios
            fig.add_trace(go.Scatter(
                x=results[0,:], 
                y=results[1,:], 
                mode='markers',
                name='Random Portfolios',
                marker=dict(
                    color=results[2,:],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Sharpe Ratio')
                )
            ))
            
            # Plot current portfolio
            fig.add_trace(go.Scatter(
                x=[current_volatility],
                y=[current_return],
                mode='markers',
                name='Current Portfolio',
                marker=dict(
                    color='red',
                    size=12,
                    line=dict(color='black', width=2)
                )
            ))
            
            # Plot optimal portfolio
            fig.add_trace(go.Scatter(
                x=[optimal_volatility],
                y=[optimal_return],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    color='green',
                    size=12,
                    symbol='star',
                    line=dict(color='black', width=2)
                )
            ))
            
            # Update layout
            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Annualized Volatility',
                yaxis_title='Annualized Return',
                showlegend=True,
                height=600,
                template='plotly_white'
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display optimal weights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimal Portfolio Weights")
                optimal_weights_df = pd.DataFrame({
                    'Asset': symbols,
                    'Optimal Weight': [f"{w*100:.2f}%" for w in optimal_weights]
                })
                st.table(optimal_weights_df)
                
                st.metric("Expected Return", f"{optimal_return*100:.2f}%")
                st.metric("Expected Volatility", f"{optimal_volatility*100:.2f}%")
                st.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
            
            with col2:
                st.subheader("Current Portfolio Weights")
                current_weights_df = pd.DataFrame({
                    'Asset': symbols,
                    'Current Weight': [f"{w*100:.2f}%" for w in current_weights]
                })
                st.table(current_weights_df)
                
                st.metric("Current Return", f"{current_return*100:.2f}%")
                st.metric("Current Volatility", f"{current_volatility*100:.2f}%")
                st.metric("Current Sharpe Ratio", f"{current_sharpe:.2f}")
                
        except Exception as e:
            st.error(f"Error calculating efficient frontier: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

# Tab 6: Portfolio Value Forecasting
with tab6:
    st.header("🔮 Portfolio Value Forecast")
    
    if not st.session_state.portfolio.assets:
        st.warning("Please add assets to your portfolio to use the forecasting tool.")
    else:
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            history_years = st.slider("Years of historical data:", 1, 10, 5)
        with col2:
            forecast_days = st.slider("Days to forecast:", 30, 365, 90)
        
        if st.button("Generate Portfolio Forecast"):
            st.session_state.show_forecast = True
            st.session_state.forecast_data = None  # Clear previous forecast data
            
        if st.session_state.get('show_forecast', False):
            try:
                # Get historical data for all assets
                end_date = datetime.now()
                start_date = end_date - timedelta(days=history_years*365)
                
                with st.spinner("Calculating historical portfolio values..."):
                    # Get historical prices for all assets
                    symbols = [asset.ticker for asset in st.session_state.portfolio.assets]
                    historical_prices_dict = st.session_state.data_manager.get_historical_prices(
                        symbols=symbols,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        interval='1d'
                    )
                    
                    if not historical_prices_dict:
                        st.error("No historical data returned. Please check your API key and try again.")
                        st.stop()
                    
                    # Check if we have any valid DataFrames
                    valid_data = False
                    price_dfs = []
                    
                    for ticker, df in historical_prices_dict.items():
                        if df is not None and not df.empty:
                            try:
                                df = df.rename(columns={'close': ticker})
                                if ticker in df.columns:  # Make sure the column was renamed successfully
                                    price_dfs.append(df[[ticker]])
                                    valid_data = True
                            except Exception as e:
                                st.warning(f"Error processing data for {ticker}: {str(e)}")
                    
                    if not valid_data:
                        st.error("No valid price data available for the selected assets.")
                        st.stop()
                    
                    if not price_dfs:
                        st.error("No valid price data available for the selected assets.")
                        st.stop()
                        
                    # Combine all price data into a single DataFrame
                    historical_prices = pd.concat(price_dfs, axis=1)
                    
                    # Calculate portfolio value over time
                    portfolio_values = pd.DataFrame(index=historical_prices.index)
                    
                    # Calculate values for each asset
                    for asset in st.session_state.portfolio.assets:
                        if asset.ticker in historical_prices.columns:
                            shares = asset.quantity
                            portfolio_values[asset.ticker] = historical_prices[asset.ticker] * shares
                    
                    # Sum up all asset values to get total portfolio value
                    portfolio_values['total'] = portfolio_values.sum(axis=1)
                    
                    # Prepare data for Prophet (only use the total portfolio value)
                    df = portfolio_values[['total']].reset_index()
                    df = df.rename(columns={'date': 'ds', 'total': 'y'})
                    
                    # Fit Prophet model
                    with st.spinner("Training forecasting model..."):
                        model = Prophet(
                            daily_seasonality=True,
                            weekly_seasonality=True,
                            yearly_seasonality=True,
                            seasonality_mode='multiplicative'
                        )
                        model.fit(df)
                        
                        # Create future dates for forecasting
                        future = model.make_future_dataframe(periods=forecast_days)
                        
                        # Generate forecast
                        forecast = model.predict(future)
                    
                    # Calculate and show only the total portfolio metrics
                    current_value = df['y'].iloc[-1]
                    forecasted_value = forecast['yhat'].iloc[-1]
                    change_pct = ((forecasted_value - current_value) / current_value) * 100
                    
                    # Display metrics in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Portfolio Value", f"${current_value:,.2f}")
                    with col2:
                        st.metric(
                            f"Forecasted Value in {forecast_days} days",
                            f"${forecasted_value:,.2f}",
                            delta=f"{change_pct:.2f}%"
                        )
                    
                    # Store forecast data in session state
                    st.session_state.forecast_data = {
                        'model': model,
                        'forecast': forecast,
                        'df': df,
                        'end_date': end_date,
                        'forecast_days': forecast_days
                    }
                    
                # Display the forecast (only once)
                if 'forecast_data' in st.session_state:
                    model = st.session_state.forecast_data['model']
                    forecast = st.session_state.forecast_data['forecast']
                    df = st.session_state.forecast_data['df']
                    end_date = st.session_state.forecast_data['end_date']
                    forecast_days = st.session_state.forecast_data['forecast_days']
                    
                    # Show forecast components
                    st.subheader("Forecast Components")
                    fig_components = plot_components_plotly(model, forecast)
                    st.plotly_chart(fig_components, use_container_width=True)
                    
                    # Show forecast plot
                    st.subheader("Portfolio Value Forecast")
                    fig_forecast = plot_plotly(model, forecast)
                    
                    # Add actual data points
                    fig_forecast.add_scatter(
                        x=df['ds'],
                        y=df['y'],
                        mode='markers',
                        name='Actual',
                        marker=dict(color='red', size=4)
                    )
                    
                    # Add confidence interval
                    fig_forecast.add_trace(go.Scatter(
                        x=pd.concat([forecast['ds'], forecast['ds']][::-1]),
                        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower']][::-1]),
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=True,
                        name='Confidence Interval'
                    ))
                    
                    fig_forecast.update_layout(
                        yaxis_title="Portfolio Value ($)",
                        xaxis_title="Date",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Show forecast table
                    st.subheader("Forecast Details")
                    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    forecast_display.columns = ['Date', 'Forecasted Value', 'Lower Bound', 'Upper Bound']
                    forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                    forecast_display[['Forecasted Value', 'Lower Bound', 'Upper Bound']] = \
                        forecast_display[['Forecasted Value', 'Lower Bound', 'Upper Bound']].round(2)
                    
                    st.dataframe(forecast_display.tail(10))
                    
                    # Add download button for forecast data
                    import uuid
                    csv = forecast_display.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f"portfolio_forecast_{end_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"download_btn_{end_date.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Truly unique key
                    )
                    
                    # Add actual data points
                    fig_forecast.add_scatter(
                        x=df['ds'],
                        y=df['y'],
                        mode='markers',
                        name='Actual',
                        marker=dict(color='red', size=4)
                    )
                    
                    # Add confidence interval
                    fig_forecast.add_trace(go.Scatter(
                        x=pd.concat([forecast['ds'], forecast['ds']][::-1]),
                        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower']][::-1]),
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=True,
                        name='Confidence Interval'
                    ))
                    
                    # Update layout
                    fig_forecast.update_layout(
                        yaxis_title="Portfolio Value ($)",
                        xaxis_title="Date",
                        hovermode='x unified'
                    )
                    
                    # Add download button for forecast data
                    csv = forecast_display.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f"portfolio_forecast_{end_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"download_btn_{end_date.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Truly unique key
                    )
                        
            except Exception as e:
                st.error(f"Error generating portfolio forecast: {str(e)}")
                import traceback
                st.text(traceback.format_exc())

# Run the app
if __name__ == "__main__":
    st.write("Smart Portfolio Analyzer is running...")
