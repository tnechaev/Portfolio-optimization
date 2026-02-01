import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.optimize import linprog

class PortfolioOptimizer:
    def __init__(self, tickers, start_date='2020-01-01', end_date=None, freq='1d', risk_free_rate=0.02):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.num_assets = len(tickers)
        
    def fetch_data(self):
        """
        Fetch historical data for all tickers and compute daily returns.
        Uses auto-adjusted prices so that 'Close' already accounts for dividends and splits.
        """
 
        # If single ticker, convert to list
        tickers_list = self.tickers if isinstance(self.tickers, list) else [self.tickers]
        
        # Create empty DataFrame to store adjusted Close
        adj_close_df = pd.DataFrame()
    
        # Loop over tickers to fetch auto-adjusted Close
        for ticker in tickers_list:
            data = yf.Ticker(ticker).history(start=self.start_date, end=self.end_date, auto_adjust=True)
            adj_close_df[ticker] = data['Close']  # 'Close' is already adjusted
    
        # Save in class
        self.data = adj_close_df
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252  # annualized
        self.cov_matrix = self.returns.cov() * 252    # annualized


    def fetch_data_eu_etfs(self):
        import yfinance as yf
        import pandas as pd
    
        tickers = self.tickers if isinstance(self.tickers, list) else [self.tickers]
        prices = pd.DataFrame()
    
        for t in tickers:
            df = yf.Ticker(t).history(start=self.start_date, end=self.end_date)
    
            # Convert timestamps to plain dates (remove timezone)
            df.index = df.index.tz_convert(None).normalize()
    
            prices[t] = df["Close"]
    
        # Now calendars align correctly
        self.data = prices.ffill().dropna()
    
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252


    
    def portfolio_performance(self, weights):
        weights = np.array(weights)
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe
    
    def negative_sharpe(self, weights):
        return -self.portfolio_performance(weights)[2]
    
    def check_sum(self, weights):
        return np.sum(weights) - 1

    def _default_bounds(self):
        return [(0, 1) for _ in range(self.num_assets)]

    def _default_constraints(self):
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def _resolve_bounds_constraints(self, bounds, constraints):
        if bounds is None:
            bounds = self._default_bounds()
        if constraints is None:
            constraints = self._default_constraints()
        return bounds, constraints

    def optimize_sharpe(self, bounds=None, constraints=None):
        #bounds = tuple((0,1) for _ in range(self.num_assets))
        #constraints = ({'type':'eq', 'fun': self.check_sum})
        bounds, constraints = self._resolve_bounds_constraints(bounds, constraints)
        init_guess = self.num_assets * [1./self.num_assets]
        result = minimize(self.negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        self.max_sharpe_weights = result.x
        self.max_sharpe_perf = self.portfolio_performance(self.max_sharpe_weights)
        return self.max_sharpe_weights, self.max_sharpe_perf


    def optimize_for_return(self, target_return, bounds=None):
        if bounds is None:
            bounds = self._default_bounds()
    
        # Feasibility check
        max_ret = np.dot(np.array([b[1] for b in bounds]), self.mean_returns)
        min_ret = np.dot(np.array([b[0] for b in bounds]), self.mean_returns)
    
        if not (min_ret <= target_return <= max_ret):
            raise ValueError("Target return infeasible under given bounds.")
    
        constraints = [
            {'type':'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type':'eq', 'fun': lambda w: np.dot(w, self.mean_returns) - target_return}
        ]
    
        init_guess = np.ones(self.num_assets) / self.num_assets
    
        result = minimize(
            lambda w: np.sqrt(w.T @ self.cov_matrix @ w),
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol':1e-12}
        )
    
        return result.x, self.portfolio_performance(result.x)

    
    def optimize_for_volatility(self, target_vol, bounds=None):
        #bounds = tuple((0,1) for _ in range(self.num_assets))
        if bounds is None:
            bounds = self._default_bounds()
        constraints = (
            {'type':'eq', 'fun': self.check_sum},
            {'type':'eq', 'fun': lambda w: np.sqrt(w.T @ self.cov_matrix @ w) - target_vol}
        )
        init_guess = self.num_assets * [1./self.num_assets]
        result = minimize(lambda w: -np.dot(w, self.mean_returns), init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x, self.portfolio_performance(result.x)
    
    def simulate_efficient_frontier(self, num_portfolios=5000):
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            ret, vol, sharpe = self.portfolio_performance(weights)
            results[:,i] = [ret, vol, sharpe]
            weights_record.append(weights)
        self.ef_results = results
        self.ef_weights = weights_record
        return results, weights_record
    
    def plot_efficient_frontier(self, user_weights=None, user_label='User Portfolio'):
        if not hasattr(self, 'ef_results'):
            self.simulate_efficient_frontier()
        ret, vol, sharpe = self.ef_results
        plt.figure(figsize=(10,6))
        plt.scatter(vol, ret, c=sharpe, cmap='viridis', marker='o', s=10, alpha=0.5)
        plt.colorbar(label='Sharpe ratio')
        # Plot max Sharpe
        max_ret, max_vol, _ = self.max_sharpe_perf
        plt.scatter(max_vol, max_ret, marker='*', color='r', s=50, label='Max Sharpe')
        # Plot user portfolio if given
        if user_weights is not None:
            u_ret, u_vol, _ = self.portfolio_performance(user_weights)
            plt.scatter(u_vol, u_ret, marker='D', color='b', s=20, label=user_label)
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.show()

    def results_table(self, weights, label="Portfolio"):
    
        weights = np.array(weights)
        ret, vol, sharpe = self.portfolio_performance(weights)
    
        df = pd.DataFrame({
            "Asset": self.tickers,
            "Weight": weights
        })
    
        #df["Weight %"] = (df["Weight"] * 100).round(2)
    
        summary = pd.DataFrame({
            "Metric": ["Expected Return", "Volatility", "Sharpe Ratio"],
            label: [round(ret,4), round(vol,4), round(sharpe,4)]
        })
    
        return df, summary


    def optimize_cvar(self, alpha=0.95, bounds=None, target_return=None):
        if bounds is None:
            bounds = self._default_bounds()
    
        R = self.returns.values   # T x N matrix
        T, N = R.shape
    
        # Decision vars: w (N), eta (1), z (T)
        num_vars = N + 1 + T
    
        # Objective: [0...0, 1, (1/((1-alpha)*T))*1...1]
        c = np.zeros(num_vars)
        c[N] = 1
        c[N+1:] = 1 / ((1-alpha) * T)
    
        A = []
        b = []
    
        # z_i >= -w·r_i - eta
        for i in range(T):
            row = np.zeros(num_vars)
            row[:N] = -R[i]
            row[N] = -1
            row[N+1+i] = -1
            A.append(row)
            b.append(0)
    
        # z_i >= 0
        for i in range(T):
            row = np.zeros(num_vars)
            row[N+1+i] = -1
            A.append(row)
            b.append(0)
    
        A = np.array(A)
        b = np.array(b)
    
        # Equality: sum w = 1
        Aeq = np.zeros((1, num_vars))
        Aeq[0, :N] = 1
        beq = np.array([1])
    
        # Optional target return
        if target_return is not None:
            row = np.zeros(num_vars)
            row[:N] = self.mean_returns
            Aeq = np.vstack([Aeq, row])
            beq = np.append(beq, target_return)
    
        # Variable bounds
        var_bounds = list(bounds) + [(None, None)] + [(0, None)] * T
    
        res = linprog(
            c, A_ub=A, b_ub=b,
            A_eq=Aeq, b_eq=beq,
            bounds=var_bounds,
            method="highs"
        )
    
        w = res.x[:N]
        perf = self.portfolio_performance(w)
    
        return w, perf, res