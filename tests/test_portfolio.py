"""Tests for src/portfolio.py"""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.portfolio import _solve_mvo, optimize_regimes, efficient_frontier


def _moments():
    """Synthetic regime moments for 3 assets."""
    mu0    = np.array([0.01, 0.007, 0.004])
    Sigma0 = np.array([[0.0014, -0.0001, -0.0001],
                       [-0.0001, 0.0008,  0.0001],
                       [-0.0001, 0.0001,  0.0013]])
    mu1    = np.array([-0.007, 0.017, 0.007])
    Sigma1 = np.array([[0.003,  0.0001, -0.0001],
                       [0.0001, 0.0028, -0.0001],
                       [-0.0001,-0.0001, 0.0027]])
    return {
        0: {'mu': mu0, 'sigma': Sigma0},
        1: {'mu': mu1, 'sigma': Sigma1},
    }


def test_min_variance_weights_sum_to_one():
    m = _moments()[0]
    w = _solve_mvo(m['mu'], m['sigma'], objective='min_variance')
    assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_min_variance_long_only():
    m = _moments()[0]
    w = _solve_mvo(m['mu'], m['sigma'], objective='min_variance', long_only=True)
    assert (w >= -1e-6).all()


def test_max_sharpe_weights_sum_to_one():
    m = _moments()[0]
    w = _solve_mvo(m['mu'], m['sigma'], objective='max_sharpe')
    assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_optimize_regimes_returns_all_objectives():
    m = _moments()
    results = optimize_regimes(m, ['Equity', 'Bonds', 'Gold'])
    for k in [0, 1]:
        for obj in ('min_variance', 'max_sharpe', 'max_return'):
            assert obj in results[k]
            w = results[k][obj].weights
            assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_efficient_frontier_length():
    m   = _moments()[0]
    n_points = 20
    frontier, weights = efficient_frontier(m['mu'], m['sigma'],
                                           ['E', 'B', 'G'], regime=0,
                                           n_points=n_points)
    assert len(frontier) == n_points
    assert weights.shape == (n_points, 3)
