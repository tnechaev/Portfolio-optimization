# portfolio_optimizer_v2_enhanced.py
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, linprog
import matplotlib.pyplot as plt
import warnings
from pandas_datareader import data as pdr
from sklearn.covariance import LedoitWolf

class PortfolioOptimizerV2:
    def __init__(self, tickers,
                 start_date='1900-01-01',
                 end_date=None,
                 freq='1d',
                 risk_free_rate=0.02,
                 inflation=0.02):
        self.tickers = list(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.risk_free_rate = risk_free_rate
        self.inflation = inflation

        self.raw_prices = None
        self.prices = None
        self.returns = None
        self.ann_arithmetic = None
        self.ann_geometric = None
        self.cov_matrix = None
        self.num_assets = len(self.tickers)

    # -----------------------
    # Data fetch & helpers
    # -----------------------
    def fetch_data(self, proxy_map=None, use_auto_adjust=True, verbose=True):
        tickers_to_fetch = []
        self.fetch_map = {}
        for t in self.tickers:
            if proxy_map and t in proxy_map:
                proxy = proxy_map[t]
                try:
                    _ = yf.Ticker(proxy).history(start=self.start_date, end=self.end_date, auto_adjust=use_auto_adjust)
                    tickers_to_fetch.append(proxy)
                    self.fetch_map[t] = proxy
                except Exception:
                    warnings.warn(f"Proxy {proxy} for {t} could not be fetched; falling back to {t}")
                    tickers_to_fetch.append(t)
                    self.fetch_map[t] = t
            else:
                tickers_to_fetch.append(t)
                self.fetch_map[t] = t

        data = yf.download(tickers=list(set(tickers_to_fetch)),
                           start=self.start_date,
                           end=self.end_date,
                           interval=self.freq,
                           auto_adjust=use_auto_adjust,
                           progress=False)

        if 'Close' in data:
            price_df = data['Close'].copy()
        else:
            price_df = data.copy()

        missing = [tk for tk in tickers_to_fetch if tk not in price_df.columns]
        if missing:
            warnings.warn(f"Missing columns after download: {missing}")

        price_df = price_df.sort_index().ffill().dropna(how='all')

        final = pd.DataFrame(index=price_df.index)
        for orig in self.tickers:
            fetched = self.fetch_map[orig]
            if fetched in price_df.columns:
                final[orig] = price_df[fetched]
            else:
                final[orig] = np.nan

        na_cols = final.columns[final.isna().all()].tolist()
        if na_cols:
            warnings.warn(f"No price data for: {na_cols}. They will be dropped.")
            final = final.drop(columns=na_cols)
            self.tickers = [t for t in self.tickers if t not in na_cols]
            self.num_assets = len(self.tickers)

        self.prices = final.ffill().dropna()
        if verbose:
            print(f"Fetched data for: {self.fetch_map}")
            print(f"Price history from {self.prices.index[0].date()} to {self.prices.index[-1].date()}")

        self.returns = self.prices.pct_change().dropna()

        if self.freq == '1d':
            self._ann_fac = 252
        elif self.freq == '1wk':
            self._ann_fac = 52
        elif self.freq == '1mo':
            self._ann_fac = 12
        else:
            self._ann_fac = 252

        self._compute_stats()

    def fetch_fred_series(self, series_id, start="1900-01-01", freq="M", transform="pct", annualize=False):
        s = pdr.DataReader(series_id, "fred", start=start).dropna()
        if freq is not None:
            s = s.resample(freq).last()
        if transform == "pct":
            s = s.pct_change()
        elif transform == "log":
            s = np.log(s).diff()
        s = s.dropna()
        s.columns = [series_id]
        if annualize:
            fac = {"M": 12, "Q": 4, "A": 1, "D": 252}.get(freq, 1)
            s = s * fac
        return s

    # -----------------------
    # Pairwise means & cov
    # -----------------------
    def pairwise_geometric_means(self):
        mu = {}
        for c in self.returns.columns:
            r = self.returns[c].dropna()
            if len(r) < 2:
                mu[c] = np.nan
            else:
                mu[c] = (1 + r).prod() ** (self._ann_fac / len(r)) - 1
        self.ann_geometric = pd.Series(mu)
        return self.ann_geometric

    def nearest_positive_definite(self, A):
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        def is_pd(X):
            try:
                _ = np.linalg.cholesky(X)
                return True
            except np.linalg.LinAlgError:
                return False

        if is_pd(A3):
            return A3

        n = A.shape[0]
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(n)
        k = 1
        while not is_pd(A3):
            min_eig = np.min(np.real(np.linalg.eigvals(A3)))
            jitter = I * (-min_eig * k**2 + spacing)
            A3 += jitter
            k += 1
            if k > 1000:
                raise RuntimeError("Failed to make matrix positive definite")
        return A3

    def pairwise_covariance(self, min_obs=1, fill_method='single_factor'):
        assets = list(self.returns.columns)
        N = len(assets)
        cov = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)

        for i_idx, i in enumerate(assets):
            for j_idx, j in enumerate(assets):
                if j_idx < i_idx:
                    cov.loc[i, j] = cov.loc[j, i]
                    continue
                pair = self.returns[[i, j]].dropna()
                n = len(pair)
                if n >= min_obs:
                    cov_val = 0.0 if n == 1 else np.cov(pair[i], pair[j], ddof=1)[0, 1]
                    cov.loc[i, j] = cov_val
                    cov.loc[j, i] = cov_val
                else:
                    cov.loc[i, j] = np.nan
                    cov.loc[j, i] = np.nan

        cov = cov * self._ann_fac

        if cov.isna().values.any():
            diag = np.diag(cov.fillna(0).values)
            var_vec = pd.Series(diag, index=assets)
            for a in assets:
                if var_vec[a] == 0 or np.isnan(var_vec[a]):
                    s = self.returns[a].dropna()
                    var_vec[a] = s.var(ddof=1) * self._ann_fac if len(s) > 1 else 1e-8

            if fill_method in ('diagonal', 'zero'):
                for i in assets:
                    for j in assets:
                        if pd.isna(cov.loc[i, j]):
                            cov.loc[i, j] = var_vec[i] if i == j else 0.0
            else:
                cors = []
                for i in assets:
                    for j in assets:
                        if i == j:
                            continue
                        if not pd.isna(cov.loc[i, j]):
                            denom = np.sqrt(var_vec[i] * var_vec[j])
                            if denom > 0:
                                cors.append(cov.loc[i, j] / denom)
                rho_bar = np.nanmean(cors) if len(cors) > 0 else 0.0
                sigma = np.sqrt(var_vec)
                for i in assets:
                    for j in assets:
                        if pd.isna(cov.loc[i, j]):
                            cov.loc[i, j] = rho_bar * sigma[i] * sigma[j]

        cov = (cov + cov.T) / 2
        try:
            _ = np.linalg.cholesky(cov.values)
            cov_pd = cov.values
        except np.linalg.LinAlgError:
            cov_pd = self.nearest_positive_definite(cov.values)

        self.cov_matrix = pd.DataFrame(cov_pd, index=assets, columns=assets)
        return self.cov_matrix

    # -----------------------
    # Compute stats
    # -----------------------
    def _compute_stats(self):
        if self.returns is None or self.returns.shape[0] == 0:
            raise RuntimeError("No returns available. Call fetch_data() first.")

        mean_periodic = self.returns.mean(axis=0)
        self.ann_arithmetic = mean_periodic * self._ann_fac

        n_periods = self.returns.shape[0]
        geo = (1 + self.returns).prod(axis=0) ** (self._ann_fac / n_periods) - 1

        self.pairwise_geometric_means()
        self.pairwise_covariance(min_obs=1, fill_method='single_factor')

        if self.ann_geometric.isna().any():
            for idx in self.ann_geometric.index[self.ann_geometric.isna()]:
                if not pd.isna(self.ann_arithmetic.get(idx, np.nan)):
                    self.ann_geometric.loc[idx] = self.ann_arithmetic.loc[idx]
                else:
                    self.ann_geometric.loc[idx] = 0.03


    #
    # --- Efficient frontier simulation & plotting ---
    #
    def simulate_efficient_frontier(self, num_portfolios=5000, seed=42):
        np.random.seed(seed)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            w = np.random.random(self.num_assets)
            w /= w.sum()
            r, s, sr = self.portfolio_performance(w)
            results[:, i] = [r, s, sr]
            weights_record.append(w)
        self.ef_results = results
        self.ef_weights = weights_record
        return results, weights_record

    def plot_efficient_frontier(self, user_weights=None, user_label='User portfolio'):
        if not hasattr(self, 'ef_results'):
            self.simulate_efficient_frontier()
        ret, vol, sharpe = self.ef_results
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(vol, ret, c=sharpe, cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(sc, label='Sharpe')
        if hasattr(self, 'max_sharpe_perf'):
            r, v, _ = self.max_sharpe_perf
            plt.scatter(v, r, marker='*', color='red', s=120, label='Max Sharpe')
        if user_weights is not None:
            ur, uv, _ = self.portfolio_performance(user_weights)
            plt.scatter(uv, ur, marker='D', color='blue', s=60, label=user_label)
        plt.xlabel('Volatility (annual)')
        plt.ylabel('Expected Return (annual)')
        plt.title('Efficient Frontier (simulated)')
        plt.legend()
        plt.show()
    
    
    # -----------------------
    # Data-driven mean shrinkage (empirical Bayes / James-Stein-ish)
    # -----------------------
    def shrink_means_empirical(self, prior=None, lam=None):
        """
        Empirical shrinkage of means toward cross-sectional mean (data-driven lambda).
        If lam is None, compute lambda from sampling variance vs between-asset variance.
        """
        sample_mu = self.get_expected_returns(method='geometric', deflate_inflation=False)
        # sample variances of the sample mean: var(r)/n_i * ann_fac^2? Approx with var(r)*ann_fac / n
        n_obs = {c: self.returns[c].dropna().shape[0] for c in self.returns.columns}
        sample_var = self.returns.var(ddof=1)
        # variance of mean estimator (annualized)
        var_mean_est = pd.Series({c: (sample_var[c] / max(1, n_obs[c])) * self._ann_fac for c in self.returns.columns})

        avg_sampling_var = var_mean_est.mean()
        between_var = sample_mu.var(ddof=1)

        if lam is None:
            # lambda = sampling_var / (between_var + sampling_var) clipped to [0,1]
            denom = between_var + avg_sampling_var
            lam_hat = float(avg_sampling_var / denom) if denom > 0 else 1.0
            lam_hat = float(np.clip(lam_hat, 0.0, 1.0))
        else:
            lam_hat = float(np.clip(lam, 0.0, 1.0))

        if prior is None:
            prior_vec = np.repeat(sample_mu.mean(), len(sample_mu))
        else:
            prior_vec = np.array(prior)
            if prior_vec.shape[0] != len(sample_mu):
                raise ValueError("Prior length mismatch with assets.")

        shrunk = lam_hat * prior_vec + (1 - lam_hat) * sample_mu.values
        self.ann_geometric = pd.Series(shrunk, index=sample_mu.index)
        return self.ann_geometric

    # -----------------------
    # Covariance shrinkage options
    # -----------------------

    def shrink_covariance_ledoit_wolf(self):
        """
        Shrink sample covariance toward Ledoit-Wolf estimator.
        Requires scikit-learn (already installed).
        """
        if self.returns is None or self.returns.shape[0] == 0:
            raise RuntimeError("No returns available. Call fetch_data() first.")

        lw = LedoitWolf()
        lw.fit(self.returns.values)
        cov_shrunk = lw.covariance_ * self._ann_fac  # annualize
        self.cov_matrix = pd.DataFrame(cov_shrunk, index=self.returns.columns, columns=self.returns.columns)
        return self.cov_matrix

    def shrink_covariance(self, delta=0.1, prior_type='single_factor'):
        """
        Backwards-compatible simple shrink. New recommended option is shrink_covariance_ledoit_wolf().
        """
        if prior_type == 'ledoit_wolf':
            return self.shrink_covariance_ledoit_wolf()

        S = self.cov_matrix.values
        N = S.shape[0]

        if prior_type == 'diagonal':
            prior = np.diag(np.diag(S))
        elif prior_type == 'single_factor':
            var = np.diag(S)
            sigma = np.sqrt(var)
            corr = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    corr[i, j] = S[i, j] / (sigma[i] * sigma[j]) if sigma[i] * sigma[j] > 0 else 0
            rho_bar = (np.sum(corr) - N) / (N * (N - 1)) if N > 1 else 0.0
            prior = np.outer(sigma, sigma) * rho_bar
            np.fill_diagonal(prior, var)
        else:
            raise ValueError("Unknown prior_type")

        shrunk = delta * prior + (1 - delta) * S
        self.cov_matrix = pd.DataFrame(shrunk, index=self.cov_matrix.index, columns=self.cov_matrix.columns)
        return self.cov_matrix

    # -----------------------
    # EWMA covariance (fast)
    # -----------------------
    def ewma_cov(self, halflife=63):
        lam = 0.5 ** (1 / halflife)
        returns_centered = self.returns - self.returns.mean()
        S = returns_centered.cov().values  # init
        for t in range(returns_centered.shape[0]):
            r = returns_centered.iloc[t].values.reshape(-1, 1)
            S = lam * S + (1 - lam) * (r @ r.T)
        self.cov_matrix = pd.DataFrame(S * self._ann_fac, index=self.returns.columns, columns=self.returns.columns)
        return self.cov_matrix

    # -----------------------
    # Portfolio metrics & classical optimizers
    # -----------------------
    def get_expected_returns(self, method='geometric', deflate_inflation=True):
        if method == 'geometric':
            mu = self.ann_geometric.copy()
        else:
            mu = self.ann_arithmetic.copy()

        if deflate_inflation and (self.inflation is not None):
            mu = mu - self.inflation
        return mu

    def portfolio_performance(self, weights, use_mu='geometric'):
        w = np.array(weights)
        mu = self.get_expected_returns(method=(use_mu or 'geometric'), deflate_inflation=False).values
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(w.T @ self.cov_matrix.values @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else np.nan
        return ret, vol, sharpe

    def negative_sharpe(self, weights):
        return -self.portfolio_performance(weights)[2]

    def check_sum(self, weights):
        return np.sum(weights) - 1.0

    def optimize_sharpe(self, bounds=None, constraints=None):
        if bounds is None:
            bounds = [(0, 1)] * self.num_assets
        if constraints is None:
            constraints = ({'type': 'eq', 'fun': self.check_sum},)
        init_guess = np.ones(self.num_assets) / self.num_assets
        result = minimize(self.negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            warnings.warn("Optimization did not converge: " + str(result.message))
        w = result.x
        perf = self.portfolio_performance(w)
        self.max_sharpe_weights = w
        self.max_sharpe_perf = perf
        return w, perf

    def minimize_variance(self, bounds=None):
        if bounds is None:
            bounds = [(0, 1)] * self.num_assets
        cons = ({'type': 'eq', 'fun': self.check_sum},)
        init = np.ones(self.num_assets) / self.num_assets
        res = minimize(lambda w: np.sqrt(w.T @ self.cov_matrix.values @ w),
                       init, method='SLSQP', bounds=bounds, constraints=cons)
        if not res.success:
            warnings.warn("Min-variance optimization failed: " + str(res.message))
        return res.x, self.portfolio_performance(res.x)

    # -----------------------
    # Risk-parity (equal risk contribution)
    # -----------------------
    def optimize_risk_parity(self, bounds=None):
        if bounds is None:
            bounds = [(0, 1)] * self.num_assets

        def rc_obj(w):
            w = np.array(w)
            cov = self.cov_matrix.values
            total_portfolio_vol = np.sqrt(w.T @ cov @ w)
            # marginal contributions
            mrc = cov @ w
            rc = w * mrc
            # desired: equal contributions
            target = total_portfolio_vol * w.sum() / len(w)
            # measure squared deviations normalized
            return np.sum((rc - (total_portfolio_vol / len(w)))**2)

        cons = ({'type': 'eq', 'fun': self.check_sum},)
        init = np.ones(self.num_assets) / self.num_assets
        res = minimize(rc_obj, init, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500})
        if not res.success:
            warnings.warn("Risk-parity optimization failed: " + str(res.message))
        return res.x, self.portfolio_performance(res.x)

    # -----------------------
    # Sharpe with turnover penalty (transaction cost control)
    # -----------------------
    def optimize_sharpe_with_turnover(self, prev_weights=None, gamma=1e-3, bounds=None):
        if bounds is None:
            bounds = [(0, 1)] * self.num_assets
        if prev_weights is None:
            prev_weights = np.zeros(self.num_assets)

        def obj(w):
            w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
            r, v, _ = self.portfolio_performance(w)
            if v <= 0 or not np.isfinite(v):
                return 1e6
            sharpe = (r - self.risk_free_rate) / v
            turnover = np.sum(np.abs(w - prev_weights))
            return -sharpe + gamma * turnover

        cons = ({'type': 'eq', 'fun': self.check_sum},)
        init = np.ones(self.num_assets) / self.num_assets
        res = minimize(obj, init, method='SLSQP', bounds=bounds, constraints=cons)
        if not res.success:
            warnings.warn("Optimize Sharpe with turnover failed: " + str(res.message))
        return res.x, self.portfolio_performance(res.x)

    # -----------------------
    # CVaR optimization (linear program)
    # -----------------------
    def optimize_cvar(self, alpha=0.95, bounds=None, target_return=None, solver='highs'):
        """
        Minimize CVaR_alpha of portfolio returns subject to weight bounds and optional target_return.
        Implementation uses linear programming formulation (Rockafellar & Uryasev).
        """
        if bounds is None:
            bounds = [(0, 1)] * self.num_assets

        R = self.returns.dropna().values  # T x N
        T, N = R.shape

        # Decision vars: w (N), eta (1), z (T)
        num_vars = N + 1 + T
        c = np.zeros(num_vars)
        c[N] = 1.0
        c[N + 1:] = 1.0 / ((1 - alpha) * T)

        A_ub = []
        b_ub = []

        # constraints: z_i >= -w^T r_i - eta  --> -w^T r_i - eta - z_i <= 0
        for t in range(T):
            row = np.zeros(num_vars)
            row[:N] = -R[t]
            row[N] = -1.0
            row[N + 1 + t] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # z_i >= 0  --> -z_i <= 0
        for t in range(T):
            row = np.zeros(num_vars)
            row[N + 1 + t] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, num_vars))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Optional target return constraint
        if target_return is not None:
            row = np.zeros(num_vars)
            # Note: using geometric or arithmetic? use ann_arithmetic for linear constraint
            row[:N] = self.ann_arithmetic.values
            A_eq = np.vstack([A_eq, row])
            b_eq = np.append(b_eq, target_return)

        # variable bounds
        var_bounds = list(bounds) + [(None, None)] + [(0, None)] * T

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method=solver)
        if not res.success:
            warnings.warn("CVaR optimization failed: " + str(res.message))
        w = res.x[:N]
        perf = self.portfolio_performance(w)
        return w, perf, res

    # -----------------------
    # Rolling backtest (walk forward)
    # -----------------------
    def rolling_backtest(self, window=252, rebalance_freq=21,
                         method='sharpe', bounds=None,
                         shrink_mean=None, shrink_cov=None,
                         gamma=1e-3):
        if bounds is None:
            bounds = [(0, 1)] * self.num_assets
    
        returns = self.returns.copy()
        num_steps = (len(returns) - window) // rebalance_freq
        if num_steps <= 0:
            raise ValueError("Not enough data for rolling backtest")
    
        portfolio_returns = []
        weights_hist = []
    
        # start from equal weights
        prev_weights = np.ones(self.num_assets) / self.num_assets
    
        for i in range(0, len(returns) - window, rebalance_freq):
            train = returns.iloc[i:i + window]
            test = returns.iloc[i + window:i + window + rebalance_freq]
    
            old_returns = self.returns
            old_mu = self.ann_geometric.copy()
            old_cov = self.cov_matrix.copy()
    
            self.returns = train
            self._compute_stats()
    
            # mean shrink
            if shrink_mean is not None:
                lam = shrink_mean.get("lam", None)
                prior = shrink_mean.get("prior", None)
                self.shrink_means_empirical(prior=prior, lam=lam)
    
            # covariance shrink
            if shrink_cov is not None:
                if shrink_cov.get("method", "") == "ledoit_wolf":
                    self.shrink_covariance_ledoit_wolf()
                else:
                    delta = shrink_cov.get("delta", 0.1)
                    ptype = shrink_cov.get("prior_type", "single_factor")
                    self.shrink_covariance(delta=delta, prior_type=ptype)
            else:
                self.cov_matrix = train.cov() * self._ann_fac
    
            # -----------------------
            # choose optimization
            # -----------------------
            if method == 'sharpe':
                w, perf = self.optimize_sharpe(bounds=bounds)
    
            elif method == 'sharpe_tc':   # <<< NEW
                w, perf = self.optimize_sharpe_with_turnover(
                    prev_weights=prev_weights,
                    gamma=gamma,
                    bounds=bounds
                )
    
            elif method == 'minvar':
                w, perf = self.minimize_variance(bounds=bounds)
    
            elif method == 'risk_parity':
                w, perf = self.optimize_risk_parity(bounds=bounds)
    
            elif method == 'cvar':
                w, perf, _ = self.optimize_cvar(
                    alpha=shrink_cov.get('cvar_alpha', 0.95) if shrink_cov else 0.95,
                    bounds=bounds,
                    target_return=None
                )
    
            else:
                w, perf = self.optimize_sharpe(bounds=bounds)
    
            # apply to test
            p_ret = test.values @ w
            portfolio_returns.append(p_ret)
            weights_hist.append(w)
    
            prev_weights = w.copy()
    
            # restore
            self.returns = old_returns
            self.ann_geometric = old_mu
            self.cov_matrix = old_cov
    
        flat = np.concatenate(portfolio_returns)
        self.bt_returns = flat
        self.bt_weights = np.vstack(weights_hist)
        return self.bt_returns, self.bt_weights


    # -----------------------
    # Backtest stats & helpers
    # -----------------------
    def backtest_stats(self):
        r = getattr(self, 'bt_returns', None)
        if r is None or len(r) == 0:
            raise ValueError("No backtest returns found. Run rolling_backtest first.")
        ann_ret = np.mean(r) * self._ann_fac
        ann_vol = np.std(r) * np.sqrt(self._ann_fac)
        sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
        cum = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        dd = (cum / peak - 1).min()
        return ann_ret, ann_vol, sharpe, dd

    def results_table(self, weights, label="Portfolio"):
        w = np.array(weights)
        r, vol, sr = self.portfolio_performance(w)
        df = pd.DataFrame({'Asset': self.tickers, 'Weight': w})
        summary = pd.DataFrame({"Metric": ["Expected Return", "Volatility", "Sharpe Ratio"],
                                label: [round(r, 4), round(vol, 4), round(sr, 4)]})
        return df, summary
