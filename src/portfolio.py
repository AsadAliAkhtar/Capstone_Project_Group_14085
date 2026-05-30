"""
Regime-conditional portfolio optimization via mean-variance (MVO).

Uses cvxpy for explicit, readable constraint specification.
Ledoit-Wolf shrinkage is applied for covariance estimation.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from sklearn.covariance import LedoitWolf

from .plots import _fmt_matrix


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Output of a single MVO solve."""
    weights    : np.ndarray
    exp_return : float
    volatility : float
    sharpe     : float
    regime     : int
    portfolio  : str
    asset_names: list

    def summary(self):
        ann = 12
        print(f"\n── {self.portfolio}  |  Regime {self.regime} ──")
        print(f"  Expected return (ann.) : {self.exp_return * ann * 100:+.2f}%")
        print(f"  Volatility     (ann.) : {self.volatility * np.sqrt(ann) * 100:.2f}%")
        print(f"  Sharpe ratio          : {self.sharpe * np.sqrt(ann):.4f}")
        print(f"  Weights:")
        for name, w in zip(self.asset_names, self.weights):
            print(f"    {name:<12}: {w:.4f}  ({w*100:.1f}%)")


# ---------------------------------------------------------------------------
# MVO core solver
# ---------------------------------------------------------------------------

def _constraints(w, n, long_only, max_weight):
    """Standard MVO constraint list."""
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if max_weight < 1.0:
        cons.append(w <= max_weight)
    return cons


def _solve_mvo(mu, Sigma, objective='min_variance',
               rf=0.0, gamma=3.0,
               long_only=True, max_weight=1.0) -> np.ndarray:
    """
    Solve a single MVO problem using cvxpy.

    Objectives
    ----------
    'min_variance'  — global minimum variance portfolio (QP)
    'max_sharpe'    — max Sharpe via Cornuejols-Tütüncü substitution (QP)
    'max_return'    — maximise mean-variance utility w'μ - (γ/2)w'Σw (QP)

    Constraints
    -----------
    Full investment (w sums to 1), long-only (w >= 0), max_weight per asset.

    Returns
    -------
    (n,) optimal weight vector; equal-weight fallback on solver failure.
    """
    n     = len(mu)
    Sigma = 0.5 * (np.array(Sigma) + np.array(Sigma).T)

    if objective == 'min_variance':
        w    = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)),
                          _constraints(w, n, long_only, max_weight))

    elif objective == 'max_sharpe':
        # Tobin substitution: y = w / (w'(μ-rf)), κ = 1 / (w'(μ-rf))
        y, kap    = cp.Variable(n), cp.Variable()
        excess_mu = mu - rf
        cons = [excess_mu @ y == 1, cp.sum(y) == kap, kap >= 0]
        if long_only:
            cons.append(y >= 0)
        if max_weight < 1.0:
            cons.append(y <= max_weight * kap)
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, Sigma)), cons)
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ('optimal', 'optimal_inaccurate') and kap.value > 1e-10:
            w_opt = np.clip(y.value / kap.value, 0 if long_only else -np.inf, max_weight)
            return w_opt / w_opt.sum()
        return np.ones(n) / n

    elif objective == 'max_return':
        w    = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(0.5 * gamma * cp.quad_form(w, Sigma) - mu @ w),
            _constraints(w, n, long_only, max_weight)
        )
    else:
        raise ValueError(f"Unknown objective '{objective}'")

    prob.solve(solver=cp.CLARABEL)
    if prob.status in ('optimal', 'optimal_inaccurate') and w.value is not None:
        w_opt = np.clip(w.value, 0 if long_only else -np.inf, max_weight)
        return w_opt / w_opt.sum()

    print(f"  Warning: solver status '{prob.status}' — returning equal weights.")
    return np.ones(n) / n


# ---------------------------------------------------------------------------
# Regime-conditional moment estimation
# ---------------------------------------------------------------------------

def estimate_regime_moments(df_train, asset_cols, state_col='state') -> dict:
    """
    Estimate state-conditional means and covariance matrices via Ledoit-Wolf.

    Parameters
    ----------
    df_train   : pd.DataFrame with state and asset columns
    asset_cols : list of str — asset return column names
    state_col  : str — name of the state column

    Returns
    -------
    dict {k: {'mu', 'sigma', 'sigma_raw', 'shrinkage', 'n_obs'}}
    """
    moments = {}
    for k in sorted(df_train[state_col].unique()):
        R_k    = df_train.loc[df_train[state_col] == k, asset_cols].values
        T_k, n = R_k.shape
        mu_k   = R_k.mean(axis=0)
        lw     = LedoitWolf().fit(R_k)

        moments[k] = {
            'mu':        mu_k,
            'sigma':     lw.covariance_,
            'sigma_raw': np.cov(R_k, rowvar=False),
            'shrinkage': lw.shrinkage_,
            'n_obs':     T_k,
        }

        print(f"\n── Regime {k}  ({T_k} observations) ──")
        print(f"  Mean returns (annualized %):")
        for col, m in zip(asset_cols, mu_k * 12 * 100):
            print(f"    {col:<12}: {m:+.2f}%")
        print(f"  Ledoit-Wolf shrinkage α : {lw.shrinkage_:.4f}")
        print(f"  Implied annual volatilities:")
        for col, v in zip(asset_cols, np.sqrt(np.diag(lw.covariance_) * 12) * 100):
            print(f"    {col:<12}: {v:.2f}%")

    return moments


def shrinkage_diagnostics(moments, asset_cols):
    """Print the difference between shrunk and raw covariance per regime."""
    print("\n── Difference between shrunk and raw covariance ──")
    for k, m in moments.items():
        diff = m['sigma'] - m['sigma_raw']
        print(f"\nRegime {k}  (α = {m['shrinkage']:.4f}):")
        print(_fmt_matrix(diff, list(asset_cols)))


# ---------------------------------------------------------------------------
# Regime-loop MVO
# ---------------------------------------------------------------------------

def optimize_regimes(moments, asset_names,
                     rf=0.0, long_only=True, max_weight=1.0) -> dict:
    """
    Solve MVO for each regime across three objectives.

    Returns
    -------
    dict {k: {'min_variance', 'max_sharpe', 'max_return'}: PortfolioResult}
    """
    results = {}
    for k, m in moments.items():
        results[k] = {}
        for obj in ('min_variance', 'max_sharpe', 'max_return'):
            w_opt   = _solve_mvo(m['mu'], m['sigma'], objective=obj,
                                  rf=rf, long_only=long_only, max_weight=max_weight)
            exp_ret = w_opt @ m['mu']
            vol     = np.sqrt(w_opt @ m['sigma'] @ w_opt)
            sharpe  = (exp_ret - rf) / vol if vol > 1e-12 else 0.0
            res = PortfolioResult(w_opt, exp_ret, vol, sharpe, k, obj, list(asset_names))
            results[k][obj] = res
            res.summary()
    return results


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier(mu, Sigma, asset_names, regime,
                       n_points=50, long_only=True, rf=0.0):
    """
    Trace the efficient frontier by sweeping risk aversion γ (log scale).

    Returns
    -------
    frontier : pd.DataFrame with columns ['volatility', 'exp_return', 'sharpe']
    weights  : (n_points, n) array of portfolio weights
    """
    gammas = np.logspace(3, -1, n_points)
    vols, rets, sharpes, ws = [], [], [], []

    for gamma in gammas:
        w   = _solve_mvo(mu, Sigma, objective='max_return',
                          gamma=gamma, long_only=long_only)
        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)
        vols.append(vol)
        rets.append(ret)
        sharpes.append((ret - rf) / vol if vol > 1e-12 else 0.0)
        ws.append(w)

    return pd.DataFrame({'volatility': vols, 'exp_return': rets,
                         'sharpe': sharpes}), np.array(ws)


# ---------------------------------------------------------------------------
# MeanVariancePortfolio — clean public API over the MVO solver
# ---------------------------------------------------------------------------

class MeanVariancePortfolio:
    """
    Markowitz mean-variance portfolio optimizer for a single regime.

    Wraps ``_solve_mvo`` in a clean, named interface and traces the
    efficient frontier via a parametric target-return sweep (more
    accurate than the γ-grid approximation in ``efficient_frontier``).

    Parameters
    ----------
    mu          : (n,) array — expected monthly returns
    Sigma       : (n,n) array — monthly covariance matrix
    asset_names : list[str]
    rf          : float — monthly risk-free rate (default 0)
    long_only   : bool — enforce w ≥ 0 (default True)
    max_weight  : float — per-asset upper bound, 1.0 = unconstrained
    regime      : int | None — regime label attached to results
    """

    def __init__(self, mu, Sigma, asset_names,
                 rf=0.0, long_only=True, max_weight=1.0, regime=None):
        self.mu          = np.asarray(mu, dtype=float)
        self.Sigma       = 0.5 * (np.asarray(Sigma) + np.asarray(Sigma).T)
        self.asset_names = list(asset_names)
        self.rf          = rf
        self.long_only   = long_only
        self.max_weight  = max_weight
        self.regime      = regime
        self.n           = len(self.mu)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _wrap(self, w, label):
        """Clip, normalise, compute metrics, return PortfolioResult."""
        w   = np.clip(w, 0.0 if self.long_only else -np.inf, self.max_weight)
        w  /= w.sum()
        ret = float(w @ self.mu)
        vol = float(np.sqrt(w @ self.Sigma @ w))
        sr  = (ret - self.rf) / vol if vol > 1e-12 else 0.0
        return PortfolioResult(w, ret, vol, sr, self.regime, label, self.asset_names)

    # ── named portfolios ────────────────────────────────────────────────────

    def global_min_variance(self) -> PortfolioResult:
        """Global minimum-variance (GMV) portfolio."""
        w = _solve_mvo(self.mu, self.Sigma, 'min_variance',
                       long_only=self.long_only, max_weight=self.max_weight)
        return self._wrap(w, 'GMV')

    def tangency(self) -> PortfolioResult:
        """Maximum Sharpe ratio (tangency) portfolio."""
        w = _solve_mvo(self.mu, self.Sigma, 'max_sharpe', rf=self.rf,
                       long_only=self.long_only, max_weight=self.max_weight)
        return self._wrap(w, 'Tangency')

    def max_utility(self, gamma: float = 3.0) -> PortfolioResult:
        """Mean-variance utility portfolio: max w′μ − (γ/2)w′Σw."""
        w = _solve_mvo(self.mu, self.Sigma, 'max_return', gamma=gamma,
                       long_only=self.long_only, max_weight=self.max_weight)
        return self._wrap(w, f'MaxUtility(γ={gamma})')

    def target_return(self, target_mu: float) -> PortfolioResult:
        """
        Minimum-variance portfolio subject to expected return ≥ target_mu.

        This is the standard parametric efficient-frontier solve.
        Falls back to equal weights if the target is infeasible.
        """
        w_var = cp.Variable(self.n)
        cons  = [cp.sum(w_var) == 1, self.mu @ w_var >= target_mu]
        if self.long_only:
            cons.append(w_var >= 0)
        if self.max_weight < 1.0:
            cons.append(w_var <= self.max_weight)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_var, self.Sigma)), cons)
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ('optimal', 'optimal_inaccurate') and w_var.value is not None:
            return self._wrap(w_var.value, f'TargetReturn({target_mu:.5f})')
        return self._wrap(np.ones(self.n) / self.n, 'FallbackEW')

    # ── efficient frontier ───────────────────────────────────────────────────

    def efficient_frontier(self, n_points: int = 60):
        """
        Trace the efficient frontier via parametric target-return sweep.

        Returns target returns from the GMV expected return (minimum
        achievable) up to the maximum feasible expected return, solving
        a min-variance QP at each target.

        Returns
        -------
        frontier : pd.DataFrame — annualised volatility, exp_return, sharpe
        weights  : (n_points, n) ndarray — corresponding portfolio weights
        """
        gmv    = self.global_min_variance()
        mu_min = gmv.exp_return

        # Upper bound: near-max-return via very low risk aversion
        w_max  = _solve_mvo(self.mu, self.Sigma, 'max_return', gamma=0.01,
                            long_only=self.long_only, max_weight=self.max_weight)
        mu_max = float(w_max @ self.mu)

        if mu_max <= mu_min + 1e-10:
            mu_max = mu_min + abs(mu_min) * 0.5 + 1e-6

        targets = np.linspace(mu_min, mu_max, n_points)
        vols, rets, sharpes, ws = [], [], [], []
        for tgt in targets:
            res = self.target_return(tgt)
            vols.append(res.volatility * np.sqrt(12))
            rets.append(res.exp_return * 12)
            sharpes.append(res.sharpe * np.sqrt(12))
            ws.append(res.weights)

        return (
            pd.DataFrame({'volatility': vols, 'exp_return': rets, 'sharpe': sharpes}),
            np.array(ws),
        )

    def summary_table(self, gamma: float = 3.0) -> pd.DataFrame:
        """
        Return a DataFrame comparing GMV, Tangency, and Max-Utility portfolios
        with annualised metrics and per-asset weights.
        """
        portfolios = {
            'GMV':         self.global_min_variance(),
            'Tangency':    self.tangency(),
            f'MaxUtil(γ={gamma})': self.max_utility(gamma),
        }
        rows = {}
        for label, res in portfolios.items():
            row = {
                'Ann. Return (%)':  res.exp_return * 12 * 100,
                'Ann. Volatility (%)': res.volatility * np.sqrt(12) * 100,
                'Sharpe Ratio':     res.sharpe * np.sqrt(12),
            }
            for name, w in zip(self.asset_names, res.weights):
                row[name] = w
            rows[label] = row
        return pd.DataFrame(rows).T
