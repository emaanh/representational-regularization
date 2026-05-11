"""Tests for the router feature extractor used in the CIFAR pilot."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def _try_import_router():
    try:
        from cka_ens.cifar.router import (
            extract_router_features,
            oracle_topk_targets,
            train_router,
            routed_prediction,
        )
        return extract_router_features, oracle_topk_targets, train_router, routed_prediction
    except ImportError:
        pytest.skip("scikit-learn not installed; install the [cifar] extra.")


def _toy_ensemble(M=4, N=50, C=10):
    """Toy probability tensor where each model is correct on its own slice."""
    probs = np.full((M, N, C), 1.0 / C)
    y = np.random.randint(0, C, size=N)
    for m in range(M):
        idxs = np.where(np.arange(N) % M == m)[0]
        for i in idxs:
            probs[m, i] = 0.05 / (C - 1)
            probs[m, i, y[i]] = 0.95
    probs /= probs.sum(axis=2, keepdims=True)
    return probs, y


def test_feature_shapes():
    extract_router_features, *_ = _try_import_router()
    M, N, C = 4, 30, 10
    probs = np.random.dirichlet(np.ones(C), size=(M, N))
    feats = extract_router_features(probs)
    assert feats.shape == (N, 2 * M + 2)


def test_oracle_topk_targets():
    _, oracle_topk_targets, *_ = _try_import_router()
    probs, y = _toy_ensemble()
    tgts = oracle_topk_targets(probs, y, k=2)
    assert len(tgts) == probs.shape[1]
    for i, t in enumerate(tgts):
        assert t[0] != -1, "Every sample is solvable by construction"


def test_router_trains_and_beats_random():
    extract_router_features, _, train_router, routed_prediction = _try_import_router()
    probs, y = _toy_ensemble(M=4, N=400)
    router, val_acc = train_router(probs, y, k=2)
    assert val_acc > 0.5, "Router should clearly beat random on the toy task"
    preds = routed_prediction(probs, router, tau=0.5)
    assert (preds == y).mean() > 0.7
