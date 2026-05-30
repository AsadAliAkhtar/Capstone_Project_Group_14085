"""
Backtest engines for regime-based and static allocation strategies.

Two complementary implementations are provided:

1. backtest_with_transaction_costs()  — used by strategy_comparison_backtest.ipynb.
   Takes pre-computed target-weight CSVs, applies a 1-month signal lag, and
   tracks drifted weights to compute realistic transaction costs.

2. run_backtest() + helpers — used by HMM_experiments.ipynb.
   Works directly on the test DataFrame with regime labels; useful for
   exploratory in-notebook analysis.
"""

import numpy as np
import pandas as pd

from .constants import ASSET_COLS


# ===========================================================================
# 1. Production backtest (strategy_comparison_backtest)
# ===========================================================================

def backtest_with_transaction_costs(
    target_weights: pd.DataFrame,
    asset_returns:  pd.DataFrame,
    transaction_cost_bps: int  = 10,
    apply_signal_lag:     bool = True,
    include_initial_trade_cost: bool = False,
) -> pd.DataFrame:
    """
    Run a backtest with drift-adjusted transaction costs.

    For each period:
      1. Optionally lag the signal by one month to avoid look-ahead bias.
      2. Compute drifted weights (how the previous portfolio grew).
      3. Traded notional = Σ |target - drifted|.
      4. Net return = gross return − traded_notional × bps / 10000.

    Parameters
    ----------
    target_weights          : DataFrame with ASSET_COLS columns (from model export)
    asset_returns           : DataFrame with ASSET_COLS columns
    transaction_cost_bps    : int  — cost in basis points per unit traded notional
    apply_signal_lag        : bool — shift weights forward 1 period (model strategies)
    include_initial_trade_cost : bool — whether to charge cost on first period

    Returns
    -------
    DataFrame with gross_return, net_return_after_costs, trade_notional,
    one_way_turnover, transaction_cost, and per-asset weight columns.
    """
    weights = target_weights[ASSET_COLS].copy()
    returns = asset_returns[ASSET_COLS].copy()

    if apply_signal_lag:
        weights = weights.shift(1)
    weights = weights.dropna()

    common = weights.index.intersection(returns.index)
    weights, returns = weights.loc[common], returns.loc[common]

    records          = []
    previous_target  = weights.iloc[0]

    for i, date in enumerate(common):
        current_target  = weights.loc[date]
        current_returns = returns.loc[date]

        if i == 0:
            trade_notional = current_target.abs().sum() if include_initial_trade_cost else 0.0
        else:
            prev_ret       = returns.loc[common[i - 1]]
            drifted        = previous_target * (1 + prev_ret)
            drifted        = drifted / drifted.sum()
            trade_notional = (current_target - drifted).abs().sum()

        transaction_cost = trade_notional * transaction_cost_bps / 10_000
        gross_return     = (current_target * current_returns).sum()
        net_return       = gross_return - transaction_cost

        records.append({
            "Date":                    date,
            "gross_return":            gross_return,
            "net_return_after_costs":  net_return,
            "trade_notional":          trade_notional,
            "one_way_turnover":        trade_notional / 2,
            "transaction_cost":        transaction_cost,
            "index_fund_weight":       current_target["index_fund"],
            "treasury_fund_weight":    current_target["treasury_fund"],
            "gold_fund_weight":        current_target["gold_fund"],
        })
        previous_target = current_target

    return pd.DataFrame(records).set_index("Date")


def apply_asset_level_tax_sensitivity(
    target_weights: pd.DataFrame,
    asset_returns:  pd.DataFrame,
    tax_rates:      dict,
    apply_signal_lag: bool = True,
) -> pd.DataFrame:
    """
    Estimate tax drag as a sensitivity layer on top of net returns.

    Approximates taxable gains from rebalancing sales (not full tax-lot accounting).

    Parameters
    ----------
    tax_rates : dict {asset_col: tax_rate} — e.g. {"index_fund": 0.15, ...}

    Returns
    -------
    DataFrame with tax_drag and per-asset sell-weight columns.
    """
    weights = target_weights[ASSET_COLS].copy()
    returns = asset_returns[ASSET_COLS].copy()

    if apply_signal_lag:
        weights = weights.shift(1)
    weights = weights.dropna()

    common = weights.index.intersection(returns.index)
    weights, returns = weights.loc[common], returns.loc[common]

    tax_rates_series = pd.Series(tax_rates)
    records          = []
    previous_target  = weights.iloc[0]

    for i, date in enumerate(common):
        current_target = weights.loc[date]

        if i == 0:
            sell_weights       = pd.Series(0.0, index=ASSET_COLS)
            prior_asset_returns = pd.Series(0.0, index=ASSET_COLS)
        else:
            prev_ret   = returns.loc[common[i - 1]]
            drifted    = previous_target * (1 + prev_ret)
            drifted    = drifted / drifted.sum()
            sell_weights       = (drifted - current_target).clip(lower=0)
            prior_asset_returns = prev_ret

        taxable_gain = sell_weights * prior_asset_returns.clip(lower=0)
        tax_drag     = (taxable_gain * tax_rates_series).sum()

        records.append({
            "Date":                       date,
            "tax_drag":                   tax_drag,
            "index_fund_sell_weight":     sell_weights["index_fund"],
            "treasury_fund_sell_weight":  sell_weights["treasury_fund"],
            "gold_fund_sell_weight":      sell_weights["gold_fund"],
        })
        previous_target = current_target

    return pd.DataFrame(records).set_index("Date")


# ===========================================================================
# 2. In-notebook backtest helpers (HMM_experiments)
# ===========================================================================

def dynamic_strategy_returns(df_test, asset_cols, state_col,
                              regime_weights: dict,
                              tc_bps: float = 0.0) -> pd.Series:
    """
    Compute period returns for a dynamic regime-switching strategy.

    Rebalances to regime_weights[s_t] when the regime changes.
    A flat tc_bps cost is deducted at each rebalancing event.
    """
    returns = df_test[asset_cols].values
    states  = df_test[state_col].values
    T       = len(states)

    port_rets  = np.zeros(T)
    tc_cost    = tc_bps / 10_000
    prev_state = states[0]

    for t in range(T):
        s = states[t]
        port_rets[t] = regime_weights[s] @ returns[t]
        if t > 0 and s != prev_state:
            port_rets[t] -= tc_cost
        prev_state = s

    return pd.Series(port_rets, index=df_test.index, name='HMM Dynamic')


def static_strategy_returns(df_test, asset_cols,
                             weights: np.ndarray, name: str) -> pd.Series:
    """Compute period returns for a fixed-weight strategy."""
    return pd.Series(
        df_test[asset_cols].values @ weights,
        index=df_test.index, name=name
    )


def run_backtest(df_test, asset_cols, state_col, regime_weights: dict,
                 equity_col: str, bond_col: str,
                 rf: float = 0.0, freq: int = 12,
                 confidence: float = 0.95, tc_bps: float = 0.0,
                 strategy_label: str = 'HMM Dynamic') -> dict:
    """
    Run the dynamic strategy against equal-weight and 60/40 benchmarks.

    Returns
    -------
    dict with keys 'returns', 'cum_returns', 'metrics'
    """
    from .metrics import compute_metrics

    n = len(asset_cols)
    w_ew   = np.ones(n) / n
    w_6040 = np.zeros(n)
    w_6040[asset_cols.index(equity_col)] = 0.60
    w_6040[asset_cols.index(bond_col)]   = 0.40

    r_dyn  = dynamic_strategy_returns(df_test, asset_cols, state_col,
                                       regime_weights, tc_bps=tc_bps)
    r_dyn.name = strategy_label
    r_ew   = static_strategy_returns(df_test, asset_cols, w_ew,   'Equal Weight (1/n)')
    r_6040 = static_strategy_returns(df_test, asset_cols, w_6040, '60/40')

    all_returns = pd.concat([r_dyn, r_ew, r_6040], axis=1)
    cum_returns = (1 + all_returns).cumprod()

    metrics = {col: compute_metrics(all_returns[col], rf=rf,
                                     freq=freq, confidence=confidence)
               for col in all_returns.columns}

    return {
        'returns':     all_returns,
        'cum_returns': cum_returns,
        'metrics':     pd.DataFrame(metrics).T,
    }


def print_metrics_table(metrics_df: pd.DataFrame):
    """Pretty-print the performance metrics table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    fmt = metrics_df.copy()
    for col in fmt.columns:
        if 'Sharpe' in col:
            fmt[col] = fmt[col].map('{:.3f}'.format)
        else:
            fmt[col] = fmt[col].map('{:.2f}%'.format)
    print(fmt.to_string())
    print("=" * 70)
