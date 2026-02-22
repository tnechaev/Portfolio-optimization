# portfolio_optimizer_v2_enhanced.py
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, linprog
import matplotlib.pyplot as plt
import warnings
from pandas_datareader import data as pdr
from sklearn.covariance import LedoitWolf
from statsmodels.stats.correlation_tools import cov_nearest

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

        #self.prices = final.ffill().dropna()

        self.prices = final.sort_index()
        
        if verbose:
            print(f"Fetched data for: {self.fetch_map}")
            print(f"Price history from {self.prices.index[0].date()} to {self.prices.index[-1].date()}")

        #self.returns = self.prices.pct_change().dropna()

        self.returns = self.prices.pct_change()

        if self.freq == '1d':
            self._ann_fac = 252
        elif self.freq == '1wk':
            self._ann_fac = 52
        elif self.freq == '1mo':
            self._ann_fac = 12
        else:
            self._ann_fac = 252

        #self._compute_stats()
        self._stats_ready = False
    
    def compute_stats_safe(self):
        try:
            self._compute_stats()
            self._stats_ready = True
        except Exception as e:
            print("Covariance/statistics failed:")
            print(type(e).__name__, ":", e)
            self._stats_ready = False
    


    def fetch_fred_series(self, series_id, start="1900-01-01", freq="ME", transform="pct", annualize=False):
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
            fac = {"ME": 12, "Q": 4, "A": 1, "D": 252}.get(freq, 1)
            s = s * fac
        return s



    
    def build_factors(self, factor_map, freq="ME"):
        """
        Build factor returns DataFrame from a factor_map.
        Supports FRED and Yahoo sources.
    
        factor_map example:
        {
            "growth": {"yahoo": "^990100-USD-STRD"},
            "rates":  {"fred": "GS10", "transform": "diff"},
            "infl":   {"fred": "CPIAUCSL", "transform": "pct"},
            "gold":   {"yahoo": "IAU"}
        }
    
        Returns: DataFrame indexed by date with factor columns (resampled to freq)
        """
        series = {}
    
        for name, cfg in factor_map.items():
            try:
                # ---------------- FRED ----------------
                if "fred" in cfg:
                    code = cfg["fred"]
                    s = pdr.DataReader(code, "fred", self.start_date, self.end_date)
                    # Convert to Series if DataFrame with 1 column
                    if isinstance(s, pd.DataFrame) and s.shape[1] > 1:
                        s = s.iloc[:, 0]
                    s = s.copy()  # avoid side effects
    
                    # Safe renaming
                    if isinstance(s, pd.Series):
                        s.name = name
                    elif isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                        s.columns = [name]
                        s = s.iloc[:, 0]
                    else:
                        raise RuntimeError(f"Unexpected shape for factor '{name}': {s.shape}")
    
                    if freq is not None:
                        s = s.resample(freq).last()
                    transform = cfg.get("transform", "pct")
                    if transform == "pct":
                        s = s.pct_change(fill_method=None).dropna()
                    elif transform == "log":
                        s = np.log(s).diff().dropna()
                    elif transform == "diff":
                        s = s.diff().dropna()
                    else:
                        s = s.dropna()
    
                # ---------------- Yahoo ----------------
                elif "yahoo" in cfg:
                    ticker = cfg["yahoo"]
                    df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                    if df.empty:
                        raise RuntimeError(f"yfinance returned empty for {ticker}")
    
                    # Extract Adjusted Close (or Close fallback)
                    if isinstance(df.columns, pd.MultiIndex):
                        if "Adj Close" in df.columns.get_level_values(0):
                            s = df["Adj Close"].copy()
                        elif "Close" in df.columns.get_level_values(0):
                            s = df["Close"].copy()
                        else:
                            s = df.iloc[:, 0]
                    else:
                        if "Adj Close" in df.columns:
                            s = df["Adj Close"].copy()
                        elif "Close" in df.columns:
                            s = df["Close"].copy()
                        else:
                            s = df.iloc[:, 0]
    
                    s = s.copy()
    
                    # Safe renaming
                    if isinstance(s, pd.Series):
                        s.name = name
                    elif isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                        s.columns = [name]
                        s = s.iloc[:, 0]
                    else:
                        raise RuntimeError(f"Unexpected shape for factor '{name}': {s.shape}")
    
                    if freq is not None:
                        s = s.resample(freq).last()
                    transform = cfg.get("transform", "pct")
                    if transform == "pct":
                        s = s.pct_change(fill_method=None).dropna()
                    elif transform == "log":
                        s = np.log(s).diff().dropna()
                    elif transform == "diff":
                        s = s.diff().dropna()
                    else:
                        s = s.dropna()
    
                else:
                    raise ValueError(f"No data source in factor config for '{name}'")
    
                series[name] = s
    
            except Exception as e:
                warnings.warn(f"Could not build factor '{name}': {e}. Skipping this factor.")
                continue
    
        if len(series) == 0:
            raise RuntimeError("No factors could be loaded from factor_map.")
    
        # Concatenate all factor series into a DataFrame
        factors = pd.concat(list(series.values()), axis=1)
        factors.columns = list(series.keys())
        factors = factors.dropna(how="all").dropna()
        return factors



    # -----------------------
    # Factor covariance
    # -----------------------
    def factor_covariance(self, train_returns, factors):
        from sklearn.linear_model import LinearRegression
    
        # Ensure symmetric indexing and try to align frequency:
        # If no overlap, resample train_returns to factors' frequency using compound returns.
        common = train_returns.index.intersection(factors.index)
        if len(common) < max(3, factors.shape[1] * 3):  # heuristic: need some overlap
            # resample train_returns to factor freq (fallback to month end)
            try:
                target_freq = factors.index.freq or pd.infer_freq(factors.index) or 'M'
                # convert simple returns to period returns: (1+rt).prod() - 1
                train_resampled = (1.0 + train_returns).resample(target_freq).apply(
                    lambda x: (1.0 + x).prod() - 1
                ).dropna(how='all')
                common = train_resampled.index.intersection(factors.index)
                if len(common) == 0:
                    raise RuntimeError("No overlap after resampling.")
                R = train_resampled.loc[common]
                F = factors.loc[common]
            except Exception:
                # fallback: intersect original indices (will raise below if empty)
                R = train_returns.loc[common]
                F = factors.loc[common]
        else:
            R = train_returns.loc[common]
            F = factors.loc[common]
    
        if len(common) == 0:
            raise RuntimeError("No overlapping dates between train_returns and factors. Cannot build factor covariance.")
    
        if F.shape[1] == 0:
            raise RuntimeError("Factor DataFrame has zero columns.")
    
        # Fit betas asset-by-asset
        B_list = []
        resid_vars = []
        reg = LinearRegression()
        T = F.shape[0]
        for col in R.columns:
            y = R[col].dropna()
            # align y with F
            idx = y.index.intersection(F.index)
            if len(idx) == 0:
                # asset has no overlapping observations with factors
                # set small residual variance and zero betas
                B_list.append(np.zeros(F.shape[1]))
                resid_vars.append(1e-8)
                continue
            X = F.loc[idx].values
            yy = y.loc[idx].values
            if X.shape[0] < (F.shape[1] + 1):
                # too few obs to fit reliably: fallback to OLS with ridge-like damping or zeros betas
                try:
                    reg.fit(X, yy)
                except Exception:
                    beta = np.zeros(F.shape[1])
                    resid = yy - np.zeros_like(yy)
                    var_resid = np.var(resid, ddof=1) if len(resid) > 1 else max(np.var(yy, ddof=1) if len(yy) > 1 else 1e-8, 1e-8)
                    B_list.append(beta)
                    resid_vars.append(var_resid * self._ann_fac)
                    continue
            else:
                reg.fit(X, yy)
            beta = reg.coef_.reshape(-1)
            resid = yy - reg.predict(X)
            var_resid = np.var(resid, ddof=1) if len(resid) > 1 else max(np.var(yy, ddof=1) if len(yy) > 1 else 1e-8, 1e-8)
            B_list.append(beta)
            resid_vars.append(var_resid * self._ann_fac)  # annualize residual var
    
        B = np.vstack(B_list)  # shape (N_assets, n_factors)
        Sigma_f = F.cov().values * self._ann_fac
        D = np.diag(resid_vars)
    
        Sigma = B @ Sigma_f @ B.T + D
        Sigma = (Sigma + Sigma.T) / 2.0
    
        # ensure PD by eigenvalue clipping
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma = self.nearest_positive_definite(Sigma)
    
        return pd.DataFrame(Sigma, index=R.columns, columns=R.columns)


    # -----------------------
    # Covariance diagnostics
    # -----------------------
    def covariance_diagnostics(self, raw_cov, fixed_cov, eps=1e-12):
        A = np.asarray(raw_cov)
        B = np.asarray(fixed_cov)

        frob_dist = np.linalg.norm(B - A, ord='fro')
        denom = np.linalg.norm(A, ord='fro')
        frob_rel = float(frob_dist / denom) if denom > eps else np.nan

        eig_raw = np.linalg.eigvalsh(A)
        eig_fix = np.linalg.eigvalsh(B)

        neg_raw = eig_raw[eig_raw < 0]
        pos_fix_sum = eig_fix[eig_fix > 0].sum()
        if pos_fix_sum <= eps:
            neg_ratio = np.nan
        else:
            neg_ratio = float(abs(neg_raw.sum()) / pos_fix_sum) if neg_raw.size > 0 else 0.0

        min_eig_fix = float(eig_fix.min())
        if abs(min_eig_fix) < 1e-14:
            min_eig_fix = 0.0

        return {
            "frob_relative_change": frob_rel,
            "min_eigen_raw": float(eig_raw.min()),
            "min_eigen_fixed": min_eig_fix,
            "num_negative_raw": int(np.sum(eig_raw < 0)),
            "neg_variance_ratio": neg_ratio
        }

    
    def nearest_positive_definite(self, A, method='nearest', threshold=1e-15):
        """
        Convert a symmetric covariance matrix A into the nearest PSD version
        using statsmodels.cov_nearest.
        """
        A = np.asarray(A, dtype=float)
        # ensure symmetric
        A = (A + A.T) / 2.0
        # call statsmodels utility
        cov_fixed = cov_nearest(A, method=method, threshold=threshold)
        # enforce symmetry again
        cov_fixed = (cov_fixed + cov_fixed.T) / 2
        return cov_fixed

    """
    def nearest_positive_definite(self, A, eps=1e-8):

        A = np.asarray(A, dtype=float)
        A = (A + A.T) / 2.0
        # eigh for symmetric/hermitian matrices
        vals, vecs = np.linalg.eigh(A)
        # clip eigenvalues to small positive value
        vals_clipped = np.clip(vals, a_min=eps, a_max=None)
        A_pd = (vecs @ np.diag(vals_clipped) @ vecs.T)
        # enforce symmetry
        A_pd = (A_pd + A_pd.T) / 2.0
        return A_pd
        """


    
    """
    def nearest_positive_definite(self, A):
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A3 = (B + H) / 2
        A3 = (A3 + A3.T) / 2

        def is_pd(X):
            try:
                _ = np.linalg.cholesky(X)
                return True
            except np.linalg.LinAlgError:
                return False

        if is_pd(A3):
            pass
        else:
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

        # tiny jitter to ensure strict PD
        min_eig = np.min(np.real(np.linalg.eigvals(A3)))
        if min_eig <= 0:
            A3 += (-min_eig + 1e-12) * np.eye(A3.shape[0])

        return (A3 + A3.T) / 2
    """
    
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

    def pairwise_covariance(self, min_obs=1, fill_method='single_factor', eps=1e-12):
        assets = list(self.returns.columns)
        N = len(assets)
        cov = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)
    
        # Step 1: compute raw pairwise covariance
        for i_idx, i in enumerate(assets):
            xi = self.returns[i].dropna()
            n_i = len(xi)
            
            # Diagonal: variance
            if n_i >= min_obs:
                cov.loc[i, i] = xi.var(ddof=1)
            else:
                cov.loc[i, i] = np.nan
        
            # Off-diagonal: covariance
            for j_idx in range(i_idx + 1, N):
                j = assets[j_idx]
                pair = self.returns[[i, j]].dropna()
                n_pair = len(pair)
                if n_pair >= min_obs:
                    cov_val = 0.0 if n_pair == 1 else np.cov(pair[i], pair[j], ddof=1)[0, 1]
                    cov.loc[i, j] = cov_val
                    cov.loc[j, i] = cov_val
                else:
                    cov.loc[i, j] = np.nan
                    cov.loc[j, i] = np.nan
        
        cov = cov * self._ann_fac
    
        # fill NaNs if any
        if cov.isna().values.any():
            var_vec = pd.Series({a: np.nan for a in assets})
            for a in assets:
                s = self.returns[a].dropna()
                var_vec[a] = s.var(ddof=1) * self._ann_fac if len(s) > 1 else np.nan
    
            median_var = var_vec.dropna().median() if var_vec.dropna().size > 0 else 1e-6
            floor = max(median_var * 1e-6, 1e-8)
            var_vec = var_vec.fillna(floor)
    
            # robust correlation estimate for missing entries
            cors = []
            for i in range(N):
                for j in range(i + 1, N):
                    if not pd.isna(cov.iat[i, j]):
                        denom = np.sqrt(var_vec.iloc[i] * var_vec.iloc[j])
                        if denom > 0:
                            cors.append(cov.iat[i, j] / denom)
            rho_bar = float(np.nanmean(cors)) if len(cors) > 0 else 0.0
            if not np.isfinite(rho_bar):
                rho_bar = 0.0
    
            sigma = np.sqrt(var_vec.values)
            for i in range(N):
                for j in range(N):
                    if pd.isna(cov.iat[i, j]):
                        cov.iat[i, j] = rho_bar * sigma[i] * sigma[j] if i != j else var_vec.iloc[i]
    
        cov = (cov + cov.T) / 2.0
    
        # Check eigenvalues and repair only if negative
        eigs = np.linalg.eigvalsh(cov.values)
        if eigs.min() < -eps:
            try:
                cov_pd = self.nearest_positive_definite(cov.values)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("Failed to repair covariance matrix")
        else:
            cov_pd = cov.values
    
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



    def hybrid_ledoit_pairwise(self,
                               prior_type="constant_correlation",
                               ridge=1e-6):
        """
        Hybrid Ledoit-style shrinkage:
        Pairwise covariance + data-driven delta.
        """
    
        if self.returns is None or self.returns.shape[0] == 0:
            raise RuntimeError("No returns available.")
    
        X = self.returns
        N = X.shape[1]
    
        # ---------------------------------------------------
        # 1. Pairwise covariance + sample sizes
        # ---------------------------------------------------
        S = np.zeros((N, N))
        T_ij = np.zeros((N, N))
    
        for i in range(N):
            for j in range(i, N):
                valid = X.iloc[:, [i, j]].dropna()
                Tij = len(valid)
    
                if Tij > 1:
                    cov = np.cov(valid.iloc[:,0], valid.iloc[:,1], ddof=1)[0,1]
                else:
                    cov = 0.0
    
                S[i, j] = cov
                S[j, i] = cov
                T_ij[i, j] = Tij
                T_ij[j, i] = Tij
    
        S *= self._ann_fac
    
        # Ensure symmetry
        S = 0.5 * (S + S.T)
    
        variances = np.diag(S)
        stddev = np.sqrt(np.maximum(variances, 1e-12))
    
        # ---------------------------------------------------
        # 2. Build target F
        # ---------------------------------------------------
        if prior_type == "constant_correlation":
    
            corr = S / np.outer(stddev, stddev)
            mask = ~np.eye(N, dtype=bool)
            avg_corr = np.mean(corr[mask])
    
            F = avg_corr * np.outer(stddev, stddev)
            np.fill_diagonal(F, variances)
    
        elif prior_type == "single_factor":
    
            market = X.mean(axis=1)
            betas = []
    
            for col in X.columns:
                valid = X[[col]].join(market, how="inner").dropna()
                if len(valid) < 5:
                    betas.append(0.0)
                else:
                    cov = np.cov(valid[col], valid.iloc[:,1])[0,1]
                    var_m = np.var(valid.iloc[:,1])
                    beta = cov / var_m if var_m > 1e-12 else 0.0
                    betas.append(beta)
    
            betas = np.array(betas)
            var_mkt = np.var(market) * self._ann_fac
            F = np.outer(betas, betas) * var_mkt
            np.fill_diagonal(F, variances)
    
        else:
            raise ValueError("Unknown prior_type")
    
        # ---------------------------------------------------
        # 3. Estimate numerator (estimation noise)
        # ---------------------------------------------------
        # Approximate Var(S_ij) ≈ (S_ii S_jj + S_ij^2) / T_ij
    
        var_S = np.zeros((N, N))
    
        for i in range(N):
            for j in range(N):
                Tij = max(T_ij[i, j], 1)
                var_S[i, j] = (variances[i] * variances[j] +
                               S[i, j]**2) / Tij
    
        numerator = np.sum(var_S)
    
        # ---------------------------------------------------
        # 4. Denominator: distance to target
        # ---------------------------------------------------
        diff = S - F
        denominator = np.sum(diff**2)
    
        # Avoid division issues
        if denominator < 1e-12:
            delta = 0.0
        else:
            delta = numerator / denominator
    
        delta = np.clip(delta, 0, 1)
    
        # ---------------------------------------------------
        # 5. Shrink
        # ---------------------------------------------------
        Sigma = (1 - delta) * S + delta * F
    
        # ---------------------------------------------------
        # 6. PSD repair
        # ---------------------------------------------------
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals_clipped = np.maximum(eigvals, 0)
        Sigma_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
        Sigma_psd += ridge * np.eye(N)
    
        self.cov_matrix = pd.DataFrame(
            Sigma_psd,
            index=X.columns,
            columns=X.columns
        )
    
        self.delta_estimated = delta
    
        return self.cov_matrix, delta


    
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
    # Rolling backtest (walk forward) with covariance diagnostics
    # -----------------------

    def rolling_backtest(self, window=252, rebalance_freq=21,
                         method='sharpe', bounds=None,
                         shrink_mean=None, shrink_cov=None,
                         gamma=1e-3):
    
        full_returns = self.returns.copy()
        full_assets = list(full_returns.columns)
        full_N = len(full_assets)
    
        if len(full_returns) <= window:
            raise ValueError("Not enough data for rolling backtest")
    
        portfolio_returns = []
        weights_hist = []
        self.cov_diag_history = []
    
        prev_weights_full = np.ones(full_N) / full_N
    
        for i in range(0, len(full_returns) - window, rebalance_freq):
    
            train = full_returns.iloc[i:i + window].copy()
            test = full_returns.iloc[i + window:i + window + rebalance_freq].copy()
    
            # Drop assets with no data in this window
            train = train.dropna(axis=1, how='all')
            active_assets = list(train.columns)
            N_active = len(active_assets)
    
            if N_active == 0:
                continue
    
            # Save old state
            old_returns = self.returns
            old_mu = self.ann_geometric.copy() if self.ann_geometric is not None else None
            old_cov = self.cov_matrix.copy() if self.cov_matrix is not None else None
    
            # Set rolling state
            self.returns = train
            self.num_assets = N_active
    
            # --- Mean estimation ---
            self._compute_stats()
    
            if shrink_mean is not None:
                lam = shrink_mean.get("lam", None)
                prior = shrink_mean.get("prior", None)
                self.shrink_means_empirical(prior=prior, lam=lam)
    
            # --- Covariance estimation ---
            if shrink_cov is not None:
                method_cov = shrink_cov.get("method", "")
    
                if method_cov == "hybrid_ledoit":
                    prior = shrink_cov.get("prior_type", "constant_correlation")
                    self.hybrid_ledoit_pairwise(prior_type=prior)
    
                elif method_cov == "ledoit_wolf":
                    X_rect = train.dropna(axis=0)
                    if len(X_rect) > 10:
                        self.returns = X_rect
                        self.shrink_covariance_ledoit_wolf()
                    else:
                        self.cov_matrix = train.cov() * self._ann_fac
    
                elif method_cov == "ewma":
                    X_rect = train.dropna(axis=0)
                    if len(X_rect) > 10:
                        self.returns = X_rect
                        hl = shrink_cov.get("halflife", 63)
                        self.ewma_cov(halflife=hl)
                    else:
                        self.cov_matrix = train.cov() * self._ann_fac
    
                elif method_cov == "factor":
                    factors = shrink_cov["factors"]
                    self.cov_matrix = self.factor_covariance(train, factors)
    
                elif method_cov == "manual":
                    delta = shrink_cov.get("delta", 0.2)
                    ptype = shrink_cov.get("prior_type", "single_factor")
                    self.shrink_covariance(delta=delta, prior_type=ptype)
    
                else:
                    self.cov_matrix = train.cov() * self._ann_fac
    
            else:
                self.cov_matrix = train.cov() * self._ann_fac
    
            # --- Diagnostics ---
            raw_cov = train.cov() * self._ann_fac
            fixed_cov = self.cov_matrix.copy()
    
            diag = self.covariance_diagnostics(raw_cov, fixed_cov)
            diag_record = {
                'start': train.index[0],
                'end': train.index[-1],
                **diag
            }
            self.cov_diag_history.append(diag_record)
    
            # --- Align test to active universe ---
            test_active = test[active_assets].dropna()
    
            if len(test_active) == 0:
                self.returns = old_returns
                self.ann_geometric = old_mu
                self.cov_matrix = old_cov
                self.num_assets = full_N
                continue
    
            # --- Bounds ---
            if bounds is None:
                bounds_local = [(0, 1)] * N_active
            else:
                if len(bounds) != full_N:
                    raise ValueError("Length of bounds must match full asset universe.")

                if N_active == 1:
                    bounds_local = [(1.0, 1.0)]
                else:
                    bounds_local = [
                        bounds[full_assets.index(asset)]
                        for asset in active_assets
                    ]
            
            #bounds_local = [(0, 1)] * N_active
            # Map previous full weights to active
            prev_active = np.array([
                prev_weights_full[full_assets.index(a)]
                for a in active_assets
            ])
    
            # --- Optimization ---
            if method == 'sharpe':
                w_active, _ = self.optimize_sharpe(bounds=bounds_local)
    
            elif method == 'sharpe_tc':
                w_active, _ = self.optimize_sharpe_with_turnover(
                    prev_weights=prev_active,
                    gamma=gamma,
                    bounds=bounds_local
                )
    
            elif method == 'minvar':
                w_active, _ = self.minimize_variance(bounds=bounds_local)
    
            elif method == 'risk_parity':
                w_active, _ = self.optimize_risk_parity(bounds=bounds_local)
    
            elif method == 'cvar':
                alpha = shrink_cov.get('cvar_alpha', 0.95) if shrink_cov else 0.95
                w_active, _, _ = self.optimize_cvar(
                    alpha=alpha,
                    bounds=bounds_local,
                    target_return=None
                )
    
            else:
                w_active, _ = self.optimize_sharpe(bounds=bounds_local)
    
            # --- Map active weights back to full universe ---
            w_full = np.zeros(full_N)
            for idx, asset in enumerate(active_assets):
                w_full[full_assets.index(asset)] = w_active[idx]
    
            # --- Apply to test ---
            p_ret = test_active.values @ w_active
            portfolio_returns.append(p_ret)
            weights_hist.append(w_full)
    
            prev_weights_full = w_full.copy()
    
            # --- Restore object ---
            self.returns = old_returns
            self.ann_geometric = old_mu
            self.cov_matrix = old_cov
            self.num_assets = full_N
    
        self.bt_returns = np.concatenate(portfolio_returns)
        self.bt_weights = np.vstack(weights_hist)
    
        diag_df = pd.DataFrame(self.cov_diag_history)
        numeric_diag = diag_df.drop(columns=['start', 'end'], errors='ignore')
        summary = numeric_diag.agg(['mean', 'min', 'max', 'std'])
    
        print("\nCovariance diagnostics summary across all rolling windows:")
        print(summary)
    
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
