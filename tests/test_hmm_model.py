"""Tests for src/hmm_model.py"""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hmm_model import HMMRegimeModel


def _make_data(T=200, n=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n))


def test_fit_returns_self():
    model = HMMRegimeModel(n_states=2, n_restarts=2, n_iter=50)
    X     = _make_data()
    result = model.fit(X)
    assert result is model


def test_predict_states_shape():
    model = HMMRegimeModel(n_states=2, n_restarts=2, n_iter=50)
    X     = _make_data()
    model.fit(X)
    states = model.predict_states(X)
    assert states.shape == (len(X),)
    assert set(states).issubset({0, 1})


def test_smoothed_probs_sum_to_one():
    model = HMMRegimeModel(n_states=2, n_restarts=2, n_iter=50)
    X     = _make_data()
    model.fit(X)
    probs = model.smoothed_probs(X)
    assert probs.shape == (2, len(X))
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-6)


def test_filtered_probs_sum_to_one():
    model = HMMRegimeModel(n_states=2, n_restarts=2, n_iter=50)
    X     = _make_data()
    model.fit(X)
    probs = model.filtered_probs(X)
    assert probs.shape == (2, len(X))
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-6)


def test_raises_before_fit():
    model = HMMRegimeModel()
    X     = _make_data()
    with pytest.raises(RuntimeError):
        model.predict_states(X)
