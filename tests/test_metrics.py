"""Tests for src/metrics.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import (annualized_return, annualized_volatility,
                          sharpe_ratio, sortino_ratio, max_drawdown,
                          information_ratio, summarize_performance, compute_metrics)


def _pos_returns(T=60):
    """Strictly positive return series."""
    rng = np.random.default_rng(1)
    r   = pd.Series(rng.uniform(0.001, 0.02, T))
    return r


def _neg_returns(T=60):
    """Strictly negative return series."""
    rng = np.random.default_rng(2)
    r   = pd.Series(rng.uniform(-0.02, -0.001, T))
    return r


def test_annualized_return_sign_matches_mean():
    r_pos = _pos_returns()
    r_neg = _neg_returns()
    assert annualized_return(r_pos) > 0
    assert annualized_return(r_neg) < 0


def test_sharpe_sign_matches_return():
    assert sharpe_ratio(_pos_returns()) > 0
    assert sharpe_ratio(_neg_returns()) < 0


def test_max_drawdown_non_positive():
    """Max drawdown should always be <= 0."""
    r = pd.Series(np.random.default_rng(3).standard_normal(60) * 0.03)
    assert max_drawdown(r) <= 0.0


def test_summarize_performance_has_required_keys():
    r = _pos_returns()
    s = summarize_performance(r)
    for key in ("Annualized Return", "Annualized Volatility",
                 "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"):
        assert key in s.index


def test_compute_metrics_var_positive():
    r = pd.Series(np.random.default_rng(4).standard_normal(120) * 0.04)
    m = compute_metrics(r, confidence=0.95)
    assert m["VaR 95% (%)"] >= 0
    assert m["CVaR 95% (%)"] >= m["VaR 95% (%)"]
