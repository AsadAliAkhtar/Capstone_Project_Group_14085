"""
Map inferred regimes to portfolio allocation signals.
"""

import pandas as pd


def map_regime_to_weights(
    regimes,
    mapping_spec: dict,
    date_index=None,
) -> pd.DataFrame:
    """
    Convert a regime sequence into target portfolio weights.

    Parameters
    ----------
    regimes : array-like or pd.Series
        Regime labels for each date.
    mapping_spec : dict
        Mapping from regime label to weight dictionary.
        Example:
        {
            0: {"index_fund": 0.6, "treasury_fund": 0.3, "gold_fund": 0.1},
            1: {"index_fund": 0.2, "treasury_fund": 0.5, "gold_fund": 0.3},
        }
    date_index : array-like, optional
        Dates to use as the output index. If regimes is a Series and date_index
        is not provided, the Series index is used.

    Returns
    -------
    pd.DataFrame
        Target weights indexed by date with columns:
        regime_label, index_fund, treasury_fund, gold_fund.
    """
    if isinstance(regimes, pd.Series):
        regime_series = regimes.copy()
    else:
        regime_series = pd.Series(regimes, index=date_index)

    rows = []

    for date, regime in regime_series.items():
        if regime not in mapping_spec:
            raise KeyError(f"No weight mapping found for regime {regime}")

        row = {"Date": date, "regime_label": regime}
        row.update(mapping_spec[regime])
        rows.append(row)

    weights = pd.DataFrame(rows).set_index("Date").sort_index()

    return weights