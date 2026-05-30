"""Tests for src/backtest.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backtest import (backtest_with_transaction_costs,
                           dynamic_strategy_returns, static_strategy_returns)


def _make_returns(T=60, seed=0):
    rng     = np.random.default_rng(seed)
    index   = pd.date_range("2019-01-31", periods=T, freq="ME")
    returns = pd.DataFrame(
        rng.standard_normal((T, 3)) * 0.04,
        index=index, columns=["index_fund", "treasury_fund", "gold_fund"]
    )
    return returns


def _make_weights(T=60, seed=1):
    rng    = np.random.default_rng(seed)
    index  = pd.date_range("2019-01-31", periods=T, freq="ME")
    raw    = np.abs(rng.standard_normal((T, 3)))
    raw    = raw / raw.sum(axis=1, keepdims=True)
    df     = pd.DataFrame(raw, index=index,
                          columns=["index_fund", "treasury_fund", "gold_fund"])
    df["regime_label"] = (np.arange(T) // 10) % 2
    return df


def test_backtest_output_length():
    r = _make_returns(T=60)
    w = _make_weights(T=60)
    result = backtest_with_transaction_costs(w, r, apply_signal_lag=True)
    # With a 1-period lag and an intersection, expect T-1 rows
    assert len(result) == 59


def test_net_return_le_gross():
    """Net return should never exceed gross (costs are non-negative)."""
    r = _make_returns(T=60)
    w = _make_weights(T=60)
    result = backtest_with_transaction_costs(w, r, transaction_cost_bps=10,
                                             apply_signal_lag=True)
    assert (result["net_return_after_costs"] <= result["gross_return"] + 1e-10).all()


def test_dynamic_strategy_length():
    r = _make_returns(T=60)
    r["state"] = (np.arange(60) // 10) % 2
    regime_weights = {0: np.array([0.6, 0.3, 0.1]),
                      1: np.array([0.1, 0.6, 0.3])}
    port = dynamic_strategy_returns(r, ["index_fund", "treasury_fund", "gold_fund"],
                                    "state", regime_weights)
    assert len(port) == 60


def test_static_strategy_length():
    r = _make_returns(T=60)
    w = np.array([1/3, 1/3, 1/3])
    port = static_strategy_returns(r, ["index_fund", "treasury_fund", "gold_fund"],
                                   w, "EW")
    assert len(port) == 60
