"""
Estimate hidden regimes using a Markov-Switching VAR (MSMH-VAR).
"""

import numpy as np
from .msvar import em_fit


class MSVARRegimeModel:
    """
    MSMH-VAR for macroeconomic regime detection.

    Wraps the custom em_fit() function and exposes the same interface
    as HMMRegimeModel so downstream code is model-agnostic.

    Key difference from HMM: VAR(p) consumes p observations as lags,
    so the effective sample size is T - p. Regime labels are aligned
    to dates_train[p:] in the notebooks.
    """

    def __init__(self, n_states: int = 2, p: int = 1,
                 n_restarts: int = 5, max_iter: int = 300,
                 tol: float = 1e-6, random_state: int = 42,
                 verbose: bool = True):
        self.n_states    = n_states
        self.p           = p
        self.n_restarts  = n_restarts
        self.max_iter    = max_iter
        self.tol         = tol
        self.random_state = random_state
        self.verbose     = verbose
        self._result     = None

    def fit(self, X: np.ndarray) -> "MSVARRegimeModel":
        """
        Fit MSMH-VAR via EM (Hamilton filter + Kim smoother).

        Parameters
        ----------
        X : (T, n) standardized training array

        Returns
        -------
        self
        """
        self._result = em_fit(
            data         = X,
            K            = self.n_states,
            p            = self.p,
            n_restarts   = self.n_restarts,
            max_iter     = self.max_iter,
            tol          = self.tol,
            random_state = self.random_state,
            verbose      = self.verbose,
        )
        return self

    def predict_states(self, use_smoothed: bool = True) -> np.ndarray:
        """Most-probable state sequence — shape (T - p,)."""
        self._check_fitted()
        return self._result.regime_sequence(use_smoothed=use_smoothed)

    def smoothed_probs(self) -> np.ndarray:
        """Smoothed probabilities P(S_t=k | y_{1:T}) — shape (K, T-p)."""
        self._check_fitted()
        return self._result.xi_smoothed

    def filtered_probs(self) -> np.ndarray:
        """Filtered probabilities P(S_t=k | y_{1:t}) — shape (K, T-p)."""
        self._check_fitted()
        return self._result.xi_filtered

    # Aliases for backward compatibility with notebook code
    def predict_probabilities(self) -> np.ndarray:
        return self.smoothed_probs()

    @property
    def P(self):
        """Transition matrix."""
        self._check_fitted()
        return self._result.P

    @property
    def expected_durations(self):
        self._check_fitted()
        return self._result.expected_durations

    def summary(self):
        self._check_fitted()
        return self._result.summary()

    def _check_fitted(self):
        if self._result is None:
            raise RuntimeError("Call fit() before using the model.")
