"""
Load and organize macroeconomic, asset, and target-weight data.
"""

import numpy as np
import pandas as pd

from .constants import ASSET_COLS


def load_macro_data(path="../data/macro_clean.csv") -> pd.DataFrame:
    """Load monthly macroeconomic indicators."""
    df = pd.read_csv(str(path), index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.sort_index()


def load_asset_data(path="../data/market_clean.csv") -> pd.DataFrame:
    """Load monthly asset returns (equity, bonds, gold)."""
    df = pd.read_csv(str(path), index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.sort_index()


def load_target_weights(path) -> pd.DataFrame:
    """
    Load target weights exported by a model notebook and standardize columns.

    Upstream format : Date, Regime, Equity, Bonds, Gold
    Returned format : DatetimeIndex + regime_label, index_fund, treasury_fund, gold_fund
    """
    weights = pd.read_csv(path, parse_dates=["Date"])
    rename_map = {
        "Regime": "regime_label",
        "Equity": "index_fund",
        "Bonds":  "treasury_fund",
        "Gold":   "gold_fund",
    }
    weights = weights.rename(columns=rename_map).set_index("Date").sort_index()

    # Convert percentages to decimals if weights sum to ~100
    if all(c in weights.columns for c in ASSET_COLS):
        if np.isclose(weights[ASSET_COLS].sum(axis=1).median(), 100.0, atol=1e-6):
            weights[ASSET_COLS] = weights[ASSET_COLS] / 100.0

    return weights


def validate_target_weights(weights: pd.DataFrame,
                             asset_cols=None) -> bool:
    """
    Validate a target-weight DataFrame.
    Raises ValueError on failure, returns True on success.
    """
    if asset_cols is None:
        asset_cols = ASSET_COLS

    missing = [c for c in ["regime_label"] + list(asset_cols) if c not in weights.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if weights[asset_cols].isna().any().any():
        raise ValueError("Target weights contain missing values.")

    weight_sums = weights[asset_cols].sum(axis=1)
    if not np.allclose(weight_sums, 1.0, atol=1e-6):
        bad = weights.loc[~np.isclose(weight_sums, 1.0, atol=1e-6)]
        raise ValueError(f"Rows do not sum to 1:\n{bad.head()}")

    if (weights[asset_cols] < -1e-6).any().any():
        bad = weights.loc[(weights[asset_cols] < -1e-6).any(axis=1)]
        raise ValueError(f"Negative weights found:\n{bad.head()}")

    return True
