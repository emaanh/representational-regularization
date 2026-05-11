"""NumPy reference: linear CKA + squared cosine."""
from __future__ import annotations

import numpy as np

__all__ = ["linear_cka", "cos2_similarity", "centering", "hsic_linear"]


def _flatten(X: np.ndarray) -> np.ndarray:
    """Reshape (n, *) to (n, d)."""
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim == 2:
        return X
    return X.reshape(X.shape[0], -1)


def centering(K: np.ndarray) -> np.ndarray:
    """Apply H K H."""
    mean_rows = K.mean(axis=0, keepdims=True)
    mean_cols = K.mean(axis=1, keepdims=True)
    mean_all = K.mean()
    return K - mean_rows - mean_cols + mean_all


def hsic_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear HSIC tr(H K_X H · H K_Y H)."""
    X = _flatten(X).astype(np.float64)
    Y = _flatten(Y).astype(np.float64)
    return float(np.sum(centering(X @ X.T) * centering(Y @ Y.T)))


def linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """Linear centered kernel alignment in [0, 1]."""
    xy = hsic_linear(X, Y)
    xx = hsic_linear(X, X)
    yy = hsic_linear(Y, Y)
    denom = np.sqrt(xx * yy)
    if denom < eps:
        return 0.0
    return xy / denom


def cos2_similarity(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """Squared cosine similarity between flattened X and Y."""
    x = _flatten(X).astype(np.float64).reshape(-1)
    y = _flatten(Y).astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < eps:
        return 0.0
    cos = float(np.dot(x, y)) / denom
    return cos * cos
