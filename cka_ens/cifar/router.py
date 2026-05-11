"""Logistic-regression router with top-K oracle supervision + soft-vote fallback."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[cifar]'`.") from e


def extract_router_features(probs: np.ndarray) -> np.ndarray:
    """Per-sample feature vector: per-model max-conf + margin + ensemble entropy + disagreement."""
    M, N, _ = probs.shape
    max_conf = np.max(probs, axis=2)
    top2 = np.sort(probs, axis=2)[:, :, -2:]
    margin = top2[:, :, 1] - top2[:, :, 0]
    avg = np.mean(probs, axis=0)
    entropy = -np.sum(avg * np.log(avg + 1e-8), axis=1)
    disagreement = np.std(np.argmax(probs, axis=2), axis=0)
    return np.concatenate([max_conf.T, margin.T,
                           entropy[:, None], disagreement[:, None]], axis=1)


def oracle_topk_targets(probs: np.ndarray, y_true: np.ndarray, k: int = 2):
    """Up-to-k correct model indices per sample; [-1] if unsolved."""
    preds = np.argmax(probs, axis=2)
    correct = (preds == y_true)
    targets = []
    for i in range(correct.shape[1]):
        good = np.where(correct[:, i])[0]
        targets.append(good[:k] if len(good) > 0 else np.array([-1]))
    return targets


def train_router(probs: np.ndarray, y_true: np.ndarray,
                 k: int = 2, test_size: float = 0.2, seed: int = 42):
    """Fit logistic regression to top-k oracle targets; return (router, val_acc)."""
    X = extract_router_features(probs)
    targets = oracle_topk_targets(probs, y_true, k=k)
    Xs, ys = [], []
    for i, ms in enumerate(targets):
        if ms[0] == -1:
            continue
        for m in ms:
            Xs.append(X[i]); ys.append(m)
    Xs = np.asarray(Xs); ys = np.asarray(ys)
    Xtr, Xval, ytr, yval = train_test_split(Xs, ys, test_size=test_size, random_state=seed)
    router = LogisticRegression(max_iter=2000)
    router.fit(Xtr, ytr)
    return router, accuracy_score(yval, router.predict(Xval))


def routed_prediction(probs: np.ndarray, router, tau: float = 0.65) -> np.ndarray:
    """Soft-vote when confident; otherwise route to the regression's pick."""
    avg = np.mean(probs, axis=0)
    conf = np.max(avg, axis=1)
    routed = router.predict(extract_router_features(probs))
    out = np.empty(len(conf), dtype=np.int64)
    for i in range(len(conf)):
        if conf[i] >= tau:
            out[i] = int(np.argmax(avg[i]))
        else:
            out[i] = int(np.argmax(probs[int(routed[i]), i]))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--combined-probs", type=Path, required=True)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.65)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    blob = np.load(args.combined_probs)
    router, val_acc = train_router(blob["probs"], blob["y"], k=args.k)
    print(f"Router validation accuracy: {val_acc:.4f}")
    preds = routed_prediction(blob["probs"], router, tau=args.tau)
    print(f"Routed accuracy: {float((preds == blob['y']).mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
