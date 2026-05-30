"""
Construct benchmark portfolios for comparison against model strategies.
"""

import pandas as pd


def build_constant_weights(index, equity_weight: float, bond_weight: float,
                            gold_weight: float, regime_label) -> pd.DataFrame:
    """
    Build a constant-weight DataFrame covering a given date index.

    Parameters
    ----------
    index         : DatetimeIndex — dates to cover
    equity_weight : float — allocation to index_fund
    bond_weight   : float — allocation to treasury_fund
    gold_weight   : float — allocation to gold_fund
    regime_label  : label for the regime_label column

    Returns
    -------
    pd.DataFrame with columns regime_label, index_fund, treasury_fund, gold_fund
    """
    return pd.DataFrame(
        {
            "regime_label":  regime_label,
            "index_fund":    equity_weight,
            "treasury_fund": bond_weight,
            "gold_fund":     gold_weight,
        },
        index=index,
    )


def build_equal_weight_benchmark(index) -> pd.DataFrame:
    """Equal-weight (1/3 each) across equity, bonds, gold."""
    return build_constant_weights(index, 1/3, 1/3, 1/3, "Equal_Weight")


def build_sixty_forty_benchmark(index) -> pd.DataFrame:
    """60% equity, 40% bonds, 0% gold."""
    return build_constant_weights(index, 0.60, 0.40, 0.0, "60_40")


def build_equity_only_benchmark(index) -> pd.DataFrame:
    """100% equity (buy-and-hold)."""
    return build_constant_weights(index, 1.0, 0.0, 0.0, "Buy_Hold_Equity")
