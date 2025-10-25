import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .portfolio import Portfolio

@dataclass
class SimulationResult:
    """Container for simulation results."""
    simulated_returns: np.ndarray
    confidence_intervals: Dict[float, Tuple[float, float]]
    var: Dict[float, float]  # Value at Risk at different confidence levels
    cvar: Dict[float, float]  # Conditional Value at Risk
    
    def get_summary(self) -> Dict:
        """Get a summary of the simulation results."""
        return {
            'mean_return': float(np.mean(self.simulated_returns)),
            'volatility': float(np.std(self.simulated_returns)),
            'sharpe_ratio': float(np.mean(self.simulated_returns) / np.std(self.simulated_returns)) if np.std(self.simulated_returns) > 0 else 0.0,
            'confidence_intervals': self.confidence_intervals,
            'value_at_risk': self.var,
            'conditional_value_at_risk': self.cvar
        }

class RiskSimulator:
    """
    A class to perform Monte Carlo simulations for portfolio risk analysis.
    """
    
    def __init__(self, portfolio: Portfolio, risk_free_rate: float = 0.05):
        """
        Initialize the RiskSimulator with a portfolio.
        
        Args:
            portfolio: The portfolio to analyze
            risk_free_rate: Annual risk-free rate (default: 5%)
        """
        self.portfolio = portfolio
        self.risk_free_rate = risk_free_rate
        self.simulation_results = {}
    
    def run_monte_carlo_simulation(
        self,
        num_simulations: int = 1000,
        time_horizon: int = 252,  # Trading days in a year
        confidence_levels: List[float] = None
    ) -> SimulationResult:
        """
        Run a Monte Carlo simulation for the portfolio.
        
        Args:
            num_simulations: Number of simulation paths
            time_horizon: Time horizon in days
            confidence_levels: List of confidence levels for VaR and CVaR (e.g., [0.95, 0.99])
            
        Returns:
            SimulationResult containing the simulation results
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
            
        # Get assets and check if portfolio is empty
        if not self.portfolio.assets:
            raise ValueError("Portfolio has no assets")
            
        n_assets = len(self.portfolio.assets)
        
        # Calculate weights based on current market values
        total_value = self.portfolio.get_total_value()
        weights = np.array([asset.current_value() / total_value for asset in self.portfolio.assets])
        
        # In a real implementation, you would use historical returns and covariances
        # For this example, we'll use random returns with some correlation
        np.random.seed(42)  # For reproducibility
        
        # Generate random returns (mean 0.1%, std 2% daily)
        # In practice, you would use historical data to estimate these parameters
        mean_returns = np.array([0.001] * n_assets)  # Daily returns
        cov_matrix = np.eye(n_assets) * 0.02  # Covariance matrix (simplified)
        
        # Generate correlated returns using Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails (matrix not positive definite),
            # use a diagonal covariance matrix as fallback
            L = np.diag(np.diag(cov_matrix) ** 0.5)
        
        # Generate uncorrelated random returns
        uncorrelated_returns = np.random.normal(
            loc=0,  # We'll add the mean later
            scale=1.0,  # Standard normal
            size=(num_simulations, time_horizon, n_assets)
        )
        
        # Transform to correlated returns
        # Reshape for matrix multiplication: (n_sims * time_horizon, n_assets) x (n_assets, n_assets)
        reshaped_returns = uncorrelated_returns.reshape(-1, n_assets)
        correlated_returns = np.dot(reshaped_returns, L.T)
        
        # Reshape back and add the mean
        correlated_returns = correlated_returns.reshape(num_simulations, time_horizon, n_assets)
        correlated_returns = mean_returns + correlated_returns * 0.02  # Scale by volatility
        correlated_returns = np.transpose(correlated_returns, (1, 0, 2))
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.zeros((time_horizon, num_simulations))
        for t in range(time_horizon):
            portfolio_returns[t] = np.dot(correlated_returns[t], weights)
        
        # Calculate terminal values (1 + r1) * (1 + r2) * ... * (1 + rT) - 1
        terminal_returns = np.prod(1 + portfolio_returns, axis=0) - 1
        
        # Calculate statistics
        mean_return = np.mean(terminal_returns)
        std_return = np.std(terminal_returns)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for cl in confidence_levels:
            alpha = 1 - cl
            lower = np.percentile(terminal_returns, (alpha/2) * 100)
            upper = np.percentile(terminal_returns, (1 - alpha/2) * 100)
            confidence_intervals[cl] = (float(lower), float(upper))
        
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        var = {}
        cvar = {}
        
        for cl in confidence_levels:
            var[cl] = float(np.percentile(terminal_returns, (1 - cl) * 100))
            cvar[cl] = float(terminal_returns[terminal_returns <= var[cl]].mean())
        
        # Store and return results
        result = SimulationResult(
            simulated_returns=terminal_returns,
            confidence_intervals=confidence_intervals,
            var=var,
            cvar=cvar
        )
        
        self.simulation_results[datetime.now()] = result
        return result
    
    def calculate_stress_scenarios(
        self,
        scenarios: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, Dict]:
        """
        Calculate portfolio performance under different stress scenarios.
        
        Args:
            scenarios: Dictionary of scenarios with asset returns under stress
                      Example: {
                          'market_crash': {'VTI': -0.3, 'VXUS': -0.25, ...},
                          'recession': {'VTI': -0.15, 'VXUS': -0.10, ...}
                      }
                      
        Returns:
            Dictionary with scenario results
        """
        if scenarios is None:
            # Default scenarios if none provided
            scenarios = {
                'market_crash': {asset.ticker: -0.30 for asset in self.portfolio.assets},
                'mild_recession': {asset.ticker: -0.15 for asset in self.portfolio.assets},
                'interest_rate_hike': {
                    asset.ticker: -0.10 if getattr(asset, 'asset_type', '') == 'bond' else -0.05 
                    for asset in self.portfolio.assets
                }
            }
        
        results = {}
        total_value = self.portfolio.get_total_value()
        
        for scenario_name, asset_returns in scenarios.items():
            scenario_loss = 0.0
            
            for asset in self.portfolio.assets:
                if asset.ticker in asset_returns:
                    return_pct = asset_returns[asset.ticker]
                    scenario_loss += asset.current_value() * return_pct
            
            results[scenario_name] = {
                'dollar_impact': scenario_loss,
                'percent_impact': (scenario_loss / total_value) * 100 if total_value > 0 else 0.0,
                'new_portfolio_value': total_value + scenario_loss
            }
        
        return results
    
    def estimate_risk_metrics(
        self,
        time_horizon: int = 252,
        confidence_levels: List[float] = None
    ) -> Dict:
        """
        Estimate various risk metrics for the portfolio.
        
        Args:
            time_horizon: Time horizon in days
            confidence_levels: List of confidence levels for metrics
            
        Returns:
            Dictionary with risk metrics
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
            
        # Run simulation if not already done
        if not self.simulation_results:
            self.run_monte_carlo_simulation(
                time_horizon=time_horizon,
                confidence_levels=confidence_levels
            )
        
        # Get the latest simulation result
        latest_result = next(reversed(self.simulation_results.values()))
        
        # Calculate additional metrics
        metrics = {
            'expected_return': float(np.mean(latest_result.simulated_returns)),
            'volatility': float(np.std(latest_result.simulated_returns)),
            'sharpe_ratio': (
                (np.mean(latest_result.simulated_returns) - (self.risk_free_rate * (time_horizon/252))) /
                np.std(latest_result.simulated_returns)
                if np.std(latest_result.simulated_returns) > 0 else 0.0
            ),
            'max_drawdown': float(self._calculate_max_drawdown(latest_result.simulated_returns)),
            'var': latest_result.var,
            'cvar': latest_result.cvar,
            'confidence_intervals': latest_result.confidence_intervals,
            'simulation_date': max(self.simulation_results.keys()).isoformat(),
            'time_horizon_days': time_horizon
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from a series of returns."""
        if len(returns) == 0:
            return 0.0
            
        cumulative = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / (1 + peak)
        return np.min(drawdowns) if len(drawdowns) > 0 else 0.0
