"""Sequential CIFAR-10 ensemble trainer (control or CKA-penalty)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[cifar]'`.") from e

from .data import get_loaders
from .model import TinyCNN, get_model_name
from ..llm.penalty import linear_cka_torch


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["control", "cka"], required=True)
    p.add_argument("--num-models", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda", dest="lam", type=float, default=3.0)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--weights-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def _device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _eval(model: nn.Module, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _load_prior(weights_path: Path, device) -> TinyCNN:
    """Load a frozen prior member to use as a CKA anchor."""
    m = TinyCNN().to(device)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_one_member(
    idx: int,
    mode: str,
    weights_dir: Path,
    train_loader,
    test_loader,
    sample_x: torch.Tensor,
    priors: List[nn.Module],
    device: torch.device,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    lam: float = 3.0,
    warmup: int = 2,
) -> tuple[TinyCNN, Path]:
    """Train member `idx`; save weights; return the trained model."""
    name = get_model_name(idx)
    print(f"\n==== member {name}  (mode={mode}, priors={len(priors)}) ====")
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        alpha = 0.0 if mode == "control" or epoch < warmup else lam
        model.train()
        total_loss = total_pen_log = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, cur_acts = model(x, return_acts=True)
            task_loss = F.cross_entropy(logits, y)

            pen = x.new_zeros(())
            if alpha > 0.0 and priors:
                for prior in priors:
                    with torch.no_grad():
                        _, prev_acts = prior(x, return_acts=True)
                    pair = sum(linear_cka_torch(c, p) for c, p in zip(cur_acts, prev_acts)) / len(cur_acts)
                    pen = pen + pair
                pen = pen / len(priors)

            loss = task_loss + alpha * pen
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_pen_log += float(pen) * x.size(0)
            n += x.size(0)
        val_acc = _eval(model, test_loader, device)
        print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  "
              f"pen={total_pen_log/n:.4f}  val_acc={val_acc:.4f}")

    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / f"model_{name}.pt"
    torch.save(model.state_dict(), out_path)
    return model, out_path


@torch.no_grad()
def _pairwise_cka(models: List[TinyCNN], sample_x: torch.Tensor, device) -> np.ndarray:
    n = len(models)
    mat = np.zeros((n, n))
    feats = []
    for m in models:
        m.eval()
        _, a = m(sample_x.to(device), return_acts=True)
        feats.append(a)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0; continue
            mat[i, j] = float(np.mean([
                linear_cka_torch(feats[i][k], feats[j][k]).item()
                for k in range(len(feats[i]))
            ]))
    return mat


def main(argv=None) -> int:
    args = parse_args(argv)
    device = _device(args.device)
    weights_dir = args.weights_dir or Path(
        "weights_control" if args.mode == "control" else "weights_cka"
    )
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, sample_x = get_loaders(batch_size=args.batch_size)

    members: List[TinyCNN] = []
    priors: List[nn.Module] = []
    for idx in range(args.num_models):
        name = get_model_name(idx)
        wpath = weights_dir / f"model_{name}.pt"
        if wpath.exists():
            print(f"[load] member {name}")
            members.append(_load_prior(wpath, device))
            priors.append(members[-1])
            continue
        model, _ = train_one_member(
            idx, args.mode, weights_dir, train_loader, test_loader,
            sample_x, priors, device,
            epochs=args.epochs, lr=args.lr, lam=args.lam,
            warmup=args.warmup_epochs,
        )
        for p in model.parameters():
            p.requires_grad_(False)
        priors.append(model.eval())
        members.append(model)

    mat = _pairwise_cka(members, sample_x, device)
    np.savetxt(weights_dir / "pairwise_cka.txt", mat, fmt="%.4f")
    off = mat[np.triu_indices(len(members), k=1)]
    print(f"\nMean off-diagonal CKA: {off.mean():.4f}  "
          f"(std {off.std():.4f}, min {off.min():.4f}, max {off.max():.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
