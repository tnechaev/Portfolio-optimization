# Portfolio optimization and backtesting

A Python class for portfolio construction, risk-aware optimization and rolling backtests. Supports empirical mean shrinkage, Ledoit–Wolf covariance, EWMA, CVaR, risk-parity, turnover-aware Sharpe optimization.

See demo Jupyter notebook for use examples. 

## Features

- Fetch asset prices (Yahoo Finance) and macro series (FRED)
- Pairwise geometric means and covariance estimation with handling of missing data
- Empirical shrinkage of expected returns (James–Stein)  
- Ledoit–Wolf covariance shrinkage for stable, well-conditioned covariance
- EWMA covariance for reactive risk modeling  
- Portfolio optimizers:
  - Max Sharpe ratio  
  - Min-variance  
  - Risk-parity (equal risk contribution)  
  - CVaR minimization  
  - Turnover-aware Sharpe optimization

## Backtesting

- Rolling backtest with configurable window and rebalance frequency 
- Supports all optimization methods including turnover control 
- Returns portfolio-level metrics: annualized return, volatility, Sharpe ratio and maximum drawdown

## Statistics & Metrics

- Computes portfolio performance: expected return, volatility and Sharpe ratio  
- Generates backtest summary statistics and per-asset weight tables 
- Efficient frontier simulation and plotting

## Covariance & Mean Estimation

- Nearest PSD cov matrix calculation and diagnostics
- Pairwise covariance with positive-definite adjustment
- Ledoit–Wolf shrinkage (scikit-learn)  
- EWMA covariance with configurable halflife 
- Empirical Bayes / James-Stein style mean shrinkage towards cross-sectional average

## Optimization Methods

- `optimize_sharpe` – maximize Sharpe ratio  
- `minimize_variance` – minimize portfolio variance  
- `optimize_risk_parity` – equalize risk contribution  
- `optimize_cvar` – minimize Conditional Value at Risk (CVaR) 
- `optimize_sharpe_with_turnover` – penalize high turnover for transaction cost control


## Outputs

- Asset allocation weights and expected returns  
- Portfolio-level metrics: return, volatility, Sharpe ratio, max drawdown  
- Dataframes for easy reporting and analysis

## Dependencies

- `numpy`, `pandas`, `scipy`, `matplotlib`, `yfinance`, `pandas_datareader`, `scikit-learn`

