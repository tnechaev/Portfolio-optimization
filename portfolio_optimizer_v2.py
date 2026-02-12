# PortfolioOptimizerV2.py
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from pandas_datareader import data as pdr


class PortfolioOptimizerV2:
    def __init__(self, tickers,
                 start_date='1900-01-01',
                 end_date=None,
                 freq='1d',
                 risk_free_rate=0.02,
                 inflation=0.02):
        """
        tickers : list of equity/ETF tickers (strings)
        start_date / end_date: for history fetch
        freq: '1d' or '1wk' or '1mo'
        risk_free_rate: annual nominal (e.g., 0.02 = 2%)
        inflation: annual inflation to optionally deflate returns (default 2%)
        """
        self.tickers = list(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.risk_free_rate = risk_free_rate
        self.inflation = inflation

        # Data containers
        self.raw_prices = None       # as fetched (aligned)
        self.prices = None           # possibly proxy index prices
        self.returns = None          # simple returns (periodic)
        self.ann_arithmetic = None   # annualized arithmetic mean
        self.ann_geometric = None    # annualized geometric mean (CAGR)
        self.cov_matrix = None       # annualized covariance (sample or shrunk)
        self.num_assets = len(self.tickers)

    #
    # --- Data fetch & alignment ---
    #
    def fetch_data(self, proxy_map=None, use_auto_adjust=True, verbose=True):
        """
        proxy_map: dict mapping original ETF tickers -> proxy ticker (index or long-history ETF),
                   e.g. {'IWDA.AS':'^990100-USD-STRD', 'VAGF.DE':'GLAB.L', 'IGLN.L':'GC=F'}
        If proxy_map is None then fetch the tickers provided.
        The code will try to fetch the proxy; if it fails, fetch the original ticker.
        """
        tickers_to_fetch = []
        self.fetch_map = {}  # maps original -> fetched ticker

        for t in self.tickers:
            if proxy_map and t in proxy_map:
                proxy = proxy_map[t]
                # Try to fetch proxy quickly; if it fails fall back to original
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

        # download combined (this aligns calendars automatically)
        data = yf.download(tickers=list(set(tickers_to_fetch)),
                           start=self.start_date,
                           end=self.end_date,
                           interval=self.freq,
                           auto_adjust=use_auto_adjust,
                           progress=False)

        # yfinance returns multiindex for multiple tickers; focus on 'Close' or adjusted
        if 'Close' in data:
            price_df = data['Close'].copy()
        else:
            # single ticker case: data itself is the series
            price_df = data.copy()

        # Ensure we have columns for each requested fetched ticker
        missing = [tk for tk in tickers_to_fetch if tk not in price_df.columns]
        if missing:
            warnings.warn(f"Missing columns after download: {missing}")

        # Reindex and forward-fill small gaps then drop rows w/ all NaNs
        price_df = price_df.sort_index().ffill().dropna(how='all')

        # Build final prices DataFrame matching order of original tickers
        final = pd.DataFrame(index=price_df.index)
        for orig in self.tickers:
            fetched = self.fetch_map[orig]
            if fetched in price_df.columns:
                final[orig] = price_df[fetched]
            else:
                # missing -> create NaN column (will be dropped later)
                final[orig] = np.nan

        # Drop columns that are entirely NaN and warn
        na_cols = final.columns[final.isna().all()].tolist()
        if na_cols:
            warnings.warn(f"No price data for: {na_cols}. They will be dropped.")
            final = final.drop(columns=na_cols)
            # adjust tickers list
            self.tickers = [t for t in self.tickers if t not in na_cols]
            self.num_assets = len(self.tickers)

        self.prices = final.ffill().dropna()
        if verbose:
            print(f"Fetched data for: {self.fetch_map}")
            print(f"Price history from {self.prices.index[0].date()} to {self.prices.index[-1].date()}")

        # compute returns
        self.returns = self.prices.pct_change().dropna()

        # set default annualization factor depending on freq
        if self.freq == '1d':
            self._ann_fac = 252
        elif self.freq == '1wk':
            self._ann_fac = 52
        elif self.freq == '1mo':
            self._ann_fac = 12
        else:
            self._ann_fac = 252

        # compute stats
        self._compute_stats()


    def fetch_fred_series(
        self,
        series_id,
        start="1900-01-01",
        freq="M",
        transform="pct",     # "pct", "log", or None
        annualize=False
    ):
        """
        Generic FRED series loader.
    
        series_id : str
            FRED code, e.g. "DGS10"
        freq : str
            'M' (monthly), 'Q', 'A', 'D'
        transform : str
            "pct" = pct_change,
            "log" = log returns,
            None = raw levels
        annualize : bool
            Multiply returns by 12/4/252 depending on freq
        """
        s = pdr.DataReader(series_id, "fred", start=start)
        s = s.dropna()
    
        # resample
        if freq is not None:
            s = s.resample(freq).last()
    
        # transform to returns
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
    

    #
    # --- Returns & covariance calculations ---
    #


    def pairwise_geometric_means(self):
        """
        Compute per-asset geometric (CAGR) returns using
        all available data for each asset individually.
        """
        mu = {}
        for c in self.returns.columns:
            r = self.returns[c].dropna()
            if len(r) < 60:
                mu[c] = np.nan
            else:
                mu[c] = (1 + r).prod()**(self._ann_fac / len(r)) - 1
    
        self.ann_geometric = pd.Series(mu)
        return self.ann_geometric



    def nearest_positive_definite(self, A):
        """
        Higham's algorithm to find nearest positive-definite matrix.
        Returns a symmetric positive-definite matrix.
        """
        # from https://stackoverflow.com/a/63131250 (adapted)
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
    
        # ensure PD: tweak by adding small jitter until eigenvalues > 0
        def is_pd(X):
            try:
                _ = np.linalg.cholesky(X)
                return True
            except np.linalg.LinAlgError:
                return False
    
        if is_pd(A3):
            return A3
    
        # add jitter
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


    def pairwise_covariance(self, min_obs=30, fill_method='single_factor'):
        """
        Build a covariance matrix using pairwise overlapping data.
        - min_obs: only use pair if overlap >= 1 (we keep small overlaps); default 30 is conservative.
        - fill_method: 'single_factor' (use avg correlation), 'diagonal' (zero off-diags),
                       or 'zero' (set missing to 0), or 'average_corr' (fill with average corr).
        This function:
          1) computes pairwise covariances on all overlaps (no hard dropping unless zero overlap)
          2) fills any remaining NaNs with a prior
          3) enforces symmetry and positive-definiteness
        """
        assets = list(self.returns.columns)
        N = len(assets)
        cov = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)
        counts = pd.DataFrame(0, index=assets, columns=assets, dtype=int)
    
        # 1) compute pairwise covariances using available overlaps
        for i_idx, i in enumerate(assets):
            for j_idx, j in enumerate(assets):
                if j_idx < i_idx:
                    cov.loc[i, j] = cov.loc[j, i]  # reuse
                    counts.loc[i, j] = counts.loc[j, i]
                    continue
                pair = self.returns[[i, j]].dropna()
                n = len(pair)
                counts.loc[i, j] = n
                counts.loc[j, i] = n
                if n >= 1:
                    # use sample covariance (unbiased); if n==1 covariance is 0
                    if n == 1:
                        cov_val = 0.0
                    else:
                        cov_val = np.cov(pair[i], pair[j], ddof=1)[0, 1]
                    cov.loc[i, j] = cov_val
                    cov.loc[j, i] = cov_val
                else:
                    cov.loc[i, j] = np.nan
                    cov.loc[j, i] = np.nan
    
        # 2) scale to annual
        cov = cov * self._ann_fac
    
        # 3) fill NaNs with a prior depending on fill_method
        nan_mask = cov.isna()
        if nan_mask.values.any():
            # compute per-asset variances from available diagonal entries (if diagonal missing, fallback to global var)
            diag = np.diag(cov.fillna(0).values)
            var_vec = pd.Series(diag, index=assets)
            # If some diagonal are zero (or NaN), replace with sample var from returns
            for a in assets:
                if var_vec[a] == 0 or np.isnan(var_vec[a]):
                    s = self.returns[a].dropna()
                    if len(s) > 1:
                        var_vec[a] = s.var(ddof=1) * self._ann_fac
                    else:
                        var_vec[a] = 1e-8  # tiny fallback
    
            if fill_method == 'diagonal' or fill_method == 'zero':
                # zero off-diagonals => prior covariance = 0 (uncorrelated)
                for i in assets:
                    for j in assets:
                        if pd.isna(cov.loc[i, j]):
                            if i == j:
                                cov.loc[i, j] = var_vec[i]
                            else:
                                cov.loc[i, j] = 0.0
            else:
                # build single-factor / average-correlation prior
                # estimate average correlation from available pairs
                cors = []
                for i in assets:
                    for j in assets:
                        if i == j: 
                            continue
                        if not pd.isna(cov.loc[i, j]):
                            denom = np.sqrt(var_vec[i] * var_vec[j])
                            if denom > 0:
                                cors.append(cov.loc[i, j] / denom)
                if len(cors) == 0:
                    rho_bar = 0.0
                else:
                    rho_bar = np.nanmean(cors)
    
                # fill missing using rho_bar * sigma_i * sigma_j
                sigma = np.sqrt(var_vec)
                for i in assets:
                    for j in assets:
                        if pd.isna(cov.loc[i, j]):
                            cov.loc[i, j] = rho_bar * sigma[i] * sigma[j]
    
        # 4) ensure symmetry
        cov = (cov + cov.T) / 2
    
        # 5) enforce positive-definite (nearest PD)
        cov_mat = cov.values
        try:
            # If already PD, leave it
            _ = np.linalg.cholesky(cov_mat)
            pd_cov = cov_mat
        except np.linalg.LinAlgError:
            pd_cov = self.nearest_positive_definite(cov_mat)
    
        cov_pd = pd.DataFrame(pd_cov, index=assets, columns=assets)
        self.cov_matrix = cov_pd
        return self.cov_matrix
    

    

    
    def _compute_stats(self):
        """
        Compute arithmetic and geometric annualized returns and annualized cov matrix.
        """
        if self.returns is None or self.returns.shape[0] == 0:
            raise RuntimeError("No returns available. Call fetch_data() first.")

        # Arithmetic mean (periodic) -> annual arithmetic
        mean_periodic = self.returns.mean(axis=0)
        self.ann_arithmetic = mean_periodic * self._ann_fac

        # Geometric mean (CAGR) per asset
        # (1 + r1)*(1 + r2)*...^(fac / n_periods) - 1
        n_periods = self.returns.shape[0]
        geo = (1 + self.returns).prod(axis=0) ** (self._ann_fac / n_periods) - 1
        #self.ann_geometric = geo

        # Default cov (sample) annualized
        #self.cov_matrix = self.returns.cov() * self._ann_fac

        # Use pairwise estimators instead of common-window
        self.pairwise_geometric_means()
        self.pairwise_covariance(min_obs=30, fill_method='single_factor')
        
        # Safety: if any ann_geometric is NaN (too little data), fallback to arithmetic or a prior
        if self.ann_geometric.isna().any():
            # replace NaN entries with ann_arithmetic or small prior
            for idx in self.ann_geometric.index[self.ann_geometric.isna()]:
                if not pd.isna(self.ann_arithmetic.get(idx, np.nan)):
                    self.ann_geometric.loc[idx] = self.ann_arithmetic.loc[idx]
                else:
                    self.ann_geometric.loc[idx] = 0.03  # conservative fallback



    def get_expected_returns(self, method='geometric', deflate_inflation=True):
        """
        Return a vector of expected returns (annualized). Options:
         - method: 'geometric' or 'arithmetic'
         - deflate_inflation: if True, subtract inflation (so returns are real)
        """
        if method == 'geometric':
            mu = self.ann_geometric.copy()
        else:
            mu = self.ann_arithmetic.copy()

        if deflate_inflation and (self.inflation is not None):
            mu = mu - self.inflation

        return mu

    def shrink_means(self, prior=None, lam=0.5):
        """
        Shrink sample means toward prior (vector same length).
        lam in [0,1] where lam=1 => full prior, lam=0 => sample.
        If prior is None use simple economic priors:
            equities ~ 6.5%, bonds ~ 2.5%, gold ~ 2.5%
        The user should pass prior as numpy array ordered like self.tickers.
        """
        sample_mu = self.get_expected_returns(method='geometric', deflate_inflation=False)
        if prior is None:
            # default flat prior equal to the sample mean mean
            avg = sample_mu.mean()
            prior_vec = np.repeat(avg, len(sample_mu))
        else:
            prior_vec = np.array(prior)
            if prior_vec.shape[0] != sample_mu.shape[0]:
                raise ValueError("Prior length mismatch with assets.")

        shrunk = lam * prior_vec + (1 - lam) * sample_mu.values
        self.ann_geometric = pd.Series(shrunk, index=sample_mu.index)
        return self.ann_geometric

    def shrink_covariance(self, delta=0.1, prior_type='diagonal'):
        """
        Simple shrinkage: cov_shrunk = delta * prior + (1-delta) * sample_cov
        prior_type: 'diagonal' (prior = diag(sample variances)), 'single_factor' (constant correlation)
        delta: shrinkage intensity 0..1
        """
        S = self.cov_matrix.values
        N = S.shape[0]

        if prior_type == 'diagonal':
            prior = np.diag(np.diag(S))
        elif prior_type == 'single_factor':
            # constant correlation prior: sigma_i * sigma_j * rho_bar
            var = np.diag(S)
            sigma = np.sqrt(var)
            # compute average off-diagonal correlation
            corr = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    corr[i, j] = S[i, j] / (sigma[i] * sigma[j]) if sigma[i]*sigma[j] > 0 else 0
            rho_bar = (np.sum(corr) - N) / (N*(N-1))
            prior = np.outer(sigma, sigma) * rho_bar
            np.fill_diagonal(prior, var)
        else:
            raise ValueError("Unknown prior_type")

        shrunk = delta * prior + (1 - delta) * S
        self.cov_matrix = pd.DataFrame(shrunk, index=self.cov_matrix.index, columns=self.cov_matrix.columns)
        return self.cov_matrix

    def ewma_cov(self, halflife=63):
        """
        Compute EWMA covariance with given halflife (in periods).
        """
        lambda_ = 0.5 ** (1 / halflife)  # conversion to decay factor per period
        returns_centered = self.returns - self.returns.mean()
        S = returns_centered.T @ (returns_centered * 0)  # placeholder
        # compute exponentially weighted covariance
        cov = returns_centered.ewm(halflife=halflife).cov(pairwise=True)
        # pandas ewm.cov returns MultiIndex; build sample covariance at last date
        last_date = returns_centered.index[-1]
        cov_last = cov.loc[last_date]
        # fill missing diagonal if needed
        self.cov_matrix = cov_last * self._ann_fac
        return self.cov_matrix

    #
    # --- Portfolio functions & optimizers (Sharpe, target return, target vol) ---
    #
    def portfolio_performance(self, weights, use_mu='geometric'):
        w = np.array(weights)
        if use_mu == 'geometric':
            mu = self.ann_geometric.values
        else:
            mu = self.ann_arithmetic.values
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
            bounds = [(0, 1) for _ in range(self.num_assets)]
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

    def optimize_for_return(self, target_return, bounds=None):
        if bounds is None:
            bounds = [(0, 1) for _ in range(self.num_assets)]

        # feasibility check
        ub = np.array([b[1] for b in bounds])
        lb = np.array([b[0] for b in bounds])
        max_ret = float(np.dot(ub, self.ann_geometric.values))
        min_ret = float(np.dot(lb, self.ann_geometric.values))
        if not (min_ret - 1e-12 <= target_return <= max_ret + 1e-12):
            raise ValueError(f"Target return {target_return:.4f} infeasible ({min_ret:.4f} .. {max_ret:.4f}).")

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, self.ann_geometric.values) - target_return}
        ]
        init = np.ones(self.num_assets) / self.num_assets
        res = minimize(lambda w: np.sqrt(w.T @ self.cov_matrix.values @ w),
                       init, method='SLSQP', bounds=bounds, constraints=constraints)
        if not res.success:
            warnings.warn("Target-return optimization failed: " + str(res.message))
        return res.x, self.portfolio_performance(res.x)

    def optimize_for_volatility(self, target_vol, bounds=None):
        if bounds is None:
            bounds = [(0, 1) for _ in range(self.num_assets)]
        constraints = [{'type': 'eq', 'fun': self.check_sum},
                       {'type': 'eq', 'fun': lambda w: np.sqrt(w.T @ self.cov_matrix.values @ w) - target_vol}]
        init = np.ones(self.num_assets) / self.num_assets
        res = minimize(lambda w: -np.dot(w, self.ann_geometric.values),
                       init, method='SLSQP', bounds=bounds, constraints=constraints)
        if not res.success:
            warnings.warn("Target-vol optimization failed: " + str(res.message))
        return res.x, self.portfolio_performance(res.x)

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

    #
    # --- Rolling backtest (walk-forward) ---
    #

    def rolling_backtest(self, window=252, rebalance_freq=21,
                         method='sharpe', bounds=None,
                         shrink_mean=None, shrink_cov=None):
    
        if bounds is None:
            bounds = [(0, 1)] * self.num_assets
    
        returns = self.returns.copy()
        num_steps = (len(returns) - window) // rebalance_freq
        if num_steps <= 0:
            raise ValueError("Not enough data for rolling backtest")
    
        portfolio_returns = []
        weights_hist = []
    
        for i in range(0, len(returns) - window, rebalance_freq):
    
            train = returns.iloc[i:i + window]
            test = returns.iloc[i + window:i + window + rebalance_freq]
    
            # backup full-sample state
            old_returns = self.returns
            old_mu = self.ann_geometric.copy()
            old_cov = self.cov_matrix.copy()
    
            # fit on training window
            self.returns = train
            self._compute_stats()
    
            # --------- DATA-DRIVEN SHRINKAGE ----------
            if shrink_mean is not None:
                lam = shrink_mean.get("lam", 0.5)
    
                if shrink_mean.get("prior") is None:
                    # empirical Bayes prior: cross-sectional mean
                    prior = np.repeat(self.ann_geometric.mean(), self.num_assets)
                else:
                    prior = shrink_mean["prior"]
    
                self.shrink_means(prior=prior, lam=lam)
    
            if shrink_cov is not None:
                delta = shrink_cov.get("delta", 0.1)
                ptype = shrink_cov.get("prior_type", "single_factor")
                self.shrink_covariance(delta=delta, prior_type=ptype)
            else:
                self.cov_matrix = train.cov() * self._ann_fac
    
            # -------- SAFE SHARPE OPTIMIZER ----------
            def safe_negative_sharpe(w):
                # clip inside bounds
                for k, (lo, hi) in enumerate(bounds):
                    w[k] = np.clip(w[k], lo, hi)
    
                if abs(w.sum() - 1) > 1e-6:
                    return 1e6
    
                r, v, _ = self.portfolio_performance(w)
                if not np.isfinite(v) or v <= 0:
                    return 1e6
    
                return -(r - self.risk_free_rate) / v
    
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
            init = np.ones(self.num_assets) / self.num_assets
    
            res = minimize(
                safe_negative_sharpe,
                init,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-9, "disp": False, "maxiter": 200}
            )
    
            if not res.success:
                warnings.warn(f"Backtest optimizer failed at step {i}: {res.message}")
    
            w = res.x
    
            # apply to test window
            p_ret = test.values @ w
            portfolio_returns.append(p_ret)
            weights_hist.append(w)
    
            # restore full-sample state
            self.returns = old_returns
            self.ann_geometric = old_mu
            self.cov_matrix = old_cov
    
        flat = np.concatenate(portfolio_returns)
        self.bt_returns = flat
        self.bt_weights = np.vstack(weights_hist)
    
        return self.bt_returns, self.bt_weights



    def backtest_stats(self):
        """
        Return annualized return, vol, sharpe, max drawdown for bt_returns
        """
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