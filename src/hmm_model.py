"""
Estimate hidden regimes using a Gaussian Hidden Markov Model.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMRegimeModel:
    """
    Gaussian HMM for macroeconomic regime detection.

    Uses multiple random restarts to avoid local optima in the EM algorithm.
    Exposes a common interface with MSVARRegimeModel so downstream code
    (portfolio optimization, backtesting) is model-agnostic.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200,
                 n_restarts: int = 10, tol: float = 1e-6,
                 random_state: int = 42):
        self.n_states    = n_states
        self.n_iter      = n_iter
        self.n_restarts  = n_restarts
        self.tol         = tol
        self.random_state = random_state
        self._model      = None

    def fit(self, X: np.ndarray) -> "HMMRegimeModel":
        """
        Fit a Gaussian HMM via Baum-Welch EM with multiple restarts.

        Parameters
        ----------
        X : (T, n) standardized training array

        Returns
        -------
        self
        """
        best_model, best_loglik = None, -np.inf

        for i in range(self.n_restarts):
            model = GaussianHMM(
                n_components    = self.n_states,
                covariance_type = 'full',
                n_iter          = self.n_iter,
                tol             = self.tol,
                init_params     = 'stmc',
                random_state    = self.random_state + i,
            )
            try:
                model.fit(X)
                loglik = model.score(X)
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_model  = model
            except Exception as e:
                print(f"Restart {i} failed: {e}")

        print(f"Best log-likelihood (per sample): {best_loglik:.4f}")
        self._model = best_model
        return self

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """Most-likely state sequence via Viterbi algorithm."""
        self._check_fitted()
        return self._model.predict(X)

    def smoothed_probs(self, X: np.ndarray) -> np.ndarray:
        """Smoothed probabilities P(S_t=k | y_{1:T}) — shape (K, T)."""
        self._check_fitted()
        return self._model.predict_proba(X).T

    def filtered_probs(self, X: np.ndarray) -> np.ndarray:
        """Filtered probabilities P(S_t=k | y_{1:t}) — shape (K, T)."""
        self._check_fitted()
        T, _ = X.shape
        K    = self.n_states

        log_emit = self._model._compute_log_likelihood(X)
        emit     = np.exp(log_emit)
        alpha    = np.zeros((T, K))

        alpha[0] = self._model.startprob_ * emit[0]
        alpha[0] /= alpha[0].sum()

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self._model.transmat_) * emit[t]
            s = alpha[t].sum()
            alpha[t] = alpha[t] / s if s > 0 else np.ones(K) / K

        return alpha.T

    def score(self, X: np.ndarray) -> float:
        """Total log-likelihood — delegates to hmmlearn GaussianHMM.score()."""
        self._check_fitted()
        return self._model.score(X)

    # Aliases for backward compatibility with notebook code
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_states() — keeps hmmlearn call-site compatible."""
        return self.predict_states(X)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        return self.smoothed_probs(X)

    @property
    def n_components(self) -> int:
        return self.n_states

    @property
    def transmat_(self):
        self._check_fitted()
        return self._model.transmat_

    @property
    def means_(self):
        self._check_fitted()
        return self._model.means_

    @property
    def covars_(self):
        self._check_fitted()
        return self._model.covars_

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Call fit() before using the model.")
