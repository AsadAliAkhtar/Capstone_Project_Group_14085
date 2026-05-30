"""
Performance metrics for tactical allocation strategies.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual metrics (match the backtest notebook's API)
# ---------------------------------------------------------------------------

def annualized_return(monthly_returns: pd.Series) -> float:
    return (1 + monthly_returns).prod() ** (12 / len(monthly_returns)) - 1


def annualized_volatility(monthly_returns: pd.Series) -> float:
    return monthly_returns.std() * np.sqrt(12)


def sharpe_ratio(monthly_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = monthly_returns - risk_free_rate / 12
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(12)


def sortino_ratio(monthly_returns: pd.Series, target_return: float = 0.0) -> float:
    downside = monthly_returns[monthly_returns < target_return]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    return (monthly_returns.mean() - target_return) / downside.std() * np.sqrt(12)


def max_drawdown(monthly_returns: pd.Series) -> float:
    cumulative  = (1 + monthly_returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative / running_max - 1).min()


def information_ratio(monthly_returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
    common = monthly_returns.index.intersection(benchmark_returns.index)
    active = monthly_returns.loc[common] - benchmark_returns.loc[common]
    if active.std() == 0:
        return np.nan
    return active.mean() / active.std() * np.sqrt(12)


def summarize_performance(monthly_returns: pd.Series,
                          benchmark_returns: pd.Series = None,
                          risk_free_rate: float = 0.0,
                          confidence: float = 0.95) -> pd.Series:
    """Return all standard metrics as a labelled Series.

    Parameters
    ----------
    monthly_returns  : net monthly return series
    benchmark_returns: benchmark series for information ratio (optional)
    risk_free_rate   : annualized risk-free rate used for Sharpe/Sortino
    confidence       : confidence level for VaR and CVaR (default 0.95)
    """
    r     = monthly_returns.dropna().values
    alpha = 1.0 - confidence
    var_threshold = np.quantile(r, alpha)
    tail = r[r <= var_threshold]
    var  = -var_threshold
    cvar = -tail.mean() if len(tail) > 0 else np.nan

    summary = {
        "Annualized Return":             annualized_return(monthly_returns),
        "Annualized Volatility":         annualized_volatility(monthly_returns),
        "Sharpe Ratio":                  sharpe_ratio(monthly_returns, risk_free_rate=risk_free_rate),
        "Sortino Ratio":                 sortino_ratio(monthly_returns),
        "Max Drawdown":                  max_drawdown(monthly_returns),
        f"VaR {int(confidence*100)}% (monthly)":  var,
        f"CVaR {int(confidence*100)}% (monthly)": cvar,
    }
    if benchmark_returns is not None:
        summary["Information Ratio vs 60/40"] = information_ratio(
            monthly_returns, benchmark_returns
        )
    return pd.Series(summary)


# ---------------------------------------------------------------------------
# Extended metric set used by the HMM in-notebook backtest
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, rf: float = 0.0,
                    freq: int = 12, confidence: float = 0.95) -> dict:
    """
    Compute annualized return, volatility, Sharpe, max drawdown, VaR, CVaR.

    Parameters
    ----------
    returns    : periodic simple returns (monthly by default)
    rf         : risk-free rate per period
    freq       : periods per year (12 for monthly)
    confidence : VaR / CVaR confidence level

    Returns
    -------
    dict of metrics (annualized where applicable, losses reported positive)
    """
    r   = returns.dropna().values
    ann_ret = (1 + r.mean()) ** freq - 1
    ann_vol = r.std() * np.sqrt(freq)
    rf_ann  = (1 + rf) ** freq - 1
    sharpe  = (ann_ret - rf_ann) / ann_vol if ann_vol > 1e-12 else 0.0

    cum_wealth  = (1 + r).cumprod()
    rolling_max = np.maximum.accumulate(cum_wealth)
    max_dd      = (cum_wealth / rolling_max - 1).min()

    alpha         = 1 - confidence
    var_threshold = np.quantile(r, alpha)
    var           = -var_threshold
    cvar          = -r[r <= var_threshold].mean()

    return {
        "Ann. Return (%)":            ann_ret * 100,
        "Ann. Volatility (%)":        ann_vol * 100,
        "Sharpe Ratio":               sharpe,
        "Max Drawdown (%)":           max_dd  * 100,
        f"VaR {int(confidence*100)}% (%)":  var  * 100,
        f"CVaR {int(confidence*100)}% (%)": cvar * 100,
    }
