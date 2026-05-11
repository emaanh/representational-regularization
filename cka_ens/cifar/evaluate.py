"""Hard / soft / oracle voting evaluator for saved CIFAR ensembles."""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from scipy import stats
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[cifar]'`.") from e

from .model import TinyCNN, get_model_name


def _device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _load_ensemble(weights_dir: Path, device, num_models: int = 50) -> List[TinyCNN]:
    models = []
    for i in range(num_models):
        wpath = weights_dir / f"model_{get_model_name(i)}.pt"
        if not wpath.exists(): break
        m = TinyCNN().to(device)
        m.load_state_dict(torch.load(wpath, map_location=device))
        m.eval()
        models.append(m)
    return models


def _compute_metrics(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """probs (M, N, C); returns hard/soft/oracle + error IoU."""
    M, N, _ = probs.shape
    preds = np.argmax(probs, axis=2)
    hard_preds, _ = stats.mode(preds, axis=0, keepdims=False)
    hard_acc = float((hard_preds == y_true).mean())
    soft_preds = np.argmax(probs.mean(axis=0), axis=1)
    soft_acc = float((soft_preds == y_true).mean())
    correct = (preds == y_true)
    oracle_acc = float(np.any(correct, axis=0).mean())
    err = [set(np.where(preds[m] != y_true)[0]) for m in range(M)]
    pairs = list(itertools.combinations(range(M), 2))
    if len(pairs) > 1000:
        pairs = random.sample(pairs, 1000)
    ious = [len(err[i] & err[j]) / len(err[i] | err[j])
            for i, j in pairs if (err[i] | err[j])]
    individual_acc = (preds == y_true).mean(axis=1)
    return {
        "num_models": M,
        "avg_single": float(individual_acc.mean()),
        "best_single": float(individual_acc.max()),
        "hard_acc": hard_acc,
        "soft_acc": soft_acc,
        "oracle_acc": oracle_acc,
        "error_iou": float(np.mean(ious)) if ious else 0.0,
    }


@torch.no_grad()
def _model_probs(model: TinyCNN, loader, device) -> np.ndarray:
    out = []
    for x, _ in loader:
        x = x.to(device)
        out.append(F.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_ensemble(weights_dir: Path, label: str,
                      loader, y_test: np.ndarray, device) -> Optional[dict]:
    print(f"\n==== {label}  ({weights_dir}) ====")
    models = _load_ensemble(weights_dir, device)
    if not models:
        print("  no models found."); return None
    print(f"  loaded {len(models)} models; running predictions ...")
    probs = np.stack([_model_probs(m, loader, device) for m in models], axis=0)
    metrics = _compute_metrics(probs, y_test)
    print(f"  avg single = {metrics['avg_single']:.2%}")
    print(f"  hard vote  = {metrics['hard_acc']:.2%}")
    print(f"  soft vote  = {metrics['soft_acc']:.2%}")
    print(f"  oracle     = {metrics['oracle_acc']:.2%}")
    print(f"  error IoU  = {metrics['error_iou']:.4f}")
    metrics["probs"] = probs
    return metrics


def _make_loader(batch_size: int):
    tx = transforms.Compose([transforms.ToTensor()])
    test = datasets.CIFAR10("./cifar10_data", train=False, download=True, transform=tx)
    y_test = np.array(test.targets)
    return DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False), y_test


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("weights_dirs", nargs="+", type=Path)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = _device()
    loader, y_test = _make_loader(args.batch_size)
    results = {}
    for wd in args.weights_dirs:
        r = evaluate_ensemble(wd, wd.name.upper(), loader, y_test, device)
        if r is not None:
            results[wd.name.upper()] = r
    if len(results) >= 2:
        combined = np.concatenate([r["probs"] for r in results.values()], axis=0)
        c = _compute_metrics(combined, y_test)
        print(f"\nCombined oracle across all ensembles: {c['oracle_acc']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
