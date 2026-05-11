"""Tests for the linear-CKA implementation (invariants + COS² failure mode)."""
from __future__ import annotations

import numpy as np
import pytest

from cka_ens.similarity import (
    cos2_similarity,
    hsic_linear,
    linear_cka,
)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def _random(n=64, d=32):
    return np.random.randn(n, d).astype(np.float64)


def _orthogonal(d):
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    return Q


def test_cka_self_is_one():
    X = _random()
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)


def test_cka_symmetric():
    X, Y = _random(), _random()
    assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-12)


def test_cka_orthogonal_invariance():
    """If we rotate features of X by an orthogonal matrix Q, CKA is unchanged."""
    n, d = 64, 32
    X = _random(n, d); Y = _random(n, d)
    Q = _orthogonal(d)
    XQ = X @ Q
    assert linear_cka(XQ, Y) == pytest.approx(linear_cka(X, Y), abs=1e-9)


def test_cka_scaling_invariance():
    """Isotropic scaling of X must not change CKA."""
    X, Y = _random(), _random()
    for c in (0.1, 2.0, -3.5, 1e3):
        assert linear_cka(c * X, Y) == pytest.approx(linear_cka(X, Y), abs=1e-9)


def test_cka_bounded_in_unit_interval():
    """For real-valued activation matrices linear CKA is in [0, 1]."""
    for _ in range(50):
        v = linear_cka(_random(), _random())
        assert -1e-9 <= v <= 1.0 + 1e-9


def test_cka_responds_to_random_vs_correlated_pairs():
    """Independent random pairs should be much less CKA-similar than correlated pairs."""
    n, d = 128, 16
    X = _random(n, d); Y = _random(n, d)
    Z = X + 0.01 * np.random.randn(n, d)
    assert linear_cka(X, Z) > linear_cka(X, Y) + 0.5


def test_cos2_is_NOT_orthogonal_invariant():
    """Squared cosine is not orthogonal-invariant (motivates using CKA)."""
    X = _random(); Y = _random()
    Q = _orthogonal(X.shape[1])
    base = cos2_similarity(X, Y)
    rotated = cos2_similarity(X @ Q, Y)

    assert abs(base - rotated) > 1e-3


def test_cka_matches_paper_summary_stats():
    """30 random matrices have much lower mean pairwise CKA than 30 correlated ones."""
    n, d = 128, 16
    diverse = [_random(n, d) for _ in range(30)]
    diverse_off = []
    for i in range(30):
        for j in range(i + 1, 30):
            diverse_off.append(linear_cka(diverse[i], diverse[j]))
    diverse_mean = float(np.mean(diverse_off))

    shared = np.random.randn(n, d) * 5.0
    correlated = [shared + 0.1 * np.random.randn(n, d) for _ in range(30)]
    correlated_off = []
    for i in range(30):
        for j in range(i + 1, 30):
            correlated_off.append(linear_cka(correlated[i], correlated[j]))
    correlated_mean = float(np.mean(correlated_off))


    assert correlated_mean > 0.9


    assert correlated_mean - diverse_mean > 0.5


def test_hsic_self_positive():
    X = _random()
    assert hsic_linear(X, X) > 0


def test_cka_handles_flat_inputs():
    """The implementation should reshape (n, *) -> (n, d) automatically."""
    X = np.random.randn(64, 8, 8, 4)
    Y = np.random.randn(64, 8, 8, 4)
    v = linear_cka(X, Y)
    assert 0.0 <= v <= 1.0


def test_cka_handles_zero_input_safely():
    X = np.zeros((16, 8))
    Y = _random(16, 8)

    assert linear_cka(X, Y) == 0.0
