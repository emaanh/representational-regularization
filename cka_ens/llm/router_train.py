"""BGE-embedding + residual-MLP router trainer (top-K expected reward)."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import Normalizer
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e


@dataclass
class RouterConfig:
    """Router training hyperparameters."""
    seed: int = 42
    epochs: int = 50
    batch_size_train: int = 64
    batch_size_eval: int = 256
    lr: float = 1e-3
    topk: int = 3
    tau_start: float = 2.0
    tau_end: float = 0.3
    ent_bonus_start: float = 0.10
    ent_bonus_end: float = 0.0
    ce_w_start: float = 0.30
    ce_w_end: float = 0.05
    warmup_epochs: int = 5
    hard_reweight_power: float = 1.0
    hard_reweight_clip: float = 10.0
    hidden: int = 256


class ResidualRouter(nn.Module):
    """2-block residual MLP."""

    def __init__(self, input_dim: int, num_experts: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_experts)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = h + F.relu(self.fc2(h))
        return self.out(h)


def _lerp(a, b, t): return a + (b - a) * t


def _exp_interp(a, b, t):
    """Exponential interpolation from a to b at fraction t."""
    return _lerp(a, b, t) if (a <= 0 or b <= 0) else a * (b / a) ** t


def _expected_reward_topk(logits, rewards, k: int, tau: float):
    """Negative top-K expected reward per sample."""
    probs = F.softmax(logits / tau, dim=1)
    vals, idx = probs.topk(k, dim=1)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    return -(vals * rewards.gather(1, idx)).sum(dim=1)


def _expected_reward_full(logits, rewards, tau: float):
    """Negative full-softmax expected reward per sample."""
    return -(F.softmax(logits / tau, dim=1) * rewards).sum(dim=1)


def _entropy(logits, tau: float = 1.0):
    """Mean predictive entropy."""
    p = F.softmax(logits / tau, dim=1)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean()


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_router(
    checkpoint_path: Path,
    out_path: Path,
    cfg: RouterConfig = RouterConfig(),
    device: str | None = None,
) -> dict:
    """Train the router; save to out_path; return summary metrics."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(cfg.seed)

    blob = torch.load(checkpoint_path)
    X = blob["X"].numpy() if isinstance(blob["X"], torch.Tensor) else blob["X"]
    Y = blob["Y_rewards"].float() if "Y_rewards" in blob else blob["Y"].float()

    XY = np.concatenate([X, Y.numpy()], axis=1)
    _, keep = np.unique(XY, axis=0, return_index=True)
    keep = np.sort(keep)
    X = X[keep]
    Y = torch.tensor(Y.numpy()[keep], dtype=torch.float32)

    X = torch.tensor(Normalizer(norm="l2").fit_transform(X), dtype=torch.float32)
    N = len(X); perm = torch.randperm(N)
    tr = perm[: int(0.8 * N)]; va = perm[int(0.8 * N): int(0.9 * N)]; te = perm[int(0.9 * N):]
    Xt, Yt = X[tr], Y[tr]; Xv, Yv = X[va], Y[va]; Xte, Yte = X[te], Y[te]

    train_loader = DataLoader(TensorDataset(Xt, Yt), batch_size=cfg.batch_size_train, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, Yv), batch_size=cfg.batch_size_eval)

    num_experts = Y.shape[1]
    router = ResidualRouter(X.shape[1], num_experts, hidden=cfg.hidden).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        router.train()
        t = epoch / max(1, (cfg.epochs - 1))
        tau = _exp_interp(cfg.tau_start, cfg.tau_end, t)
        ent_bonus = _lerp(cfg.ent_bonus_start, cfg.ent_bonus_end, t)
        ce_w = _lerp(cfg.ce_w_start, cfg.ce_w_end, t)
        use_topk = (epoch >= cfg.warmup_epochs)

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = router(bx)
            per_sample = (_expected_reward_topk(logits, by, cfg.topk, tau)
                          if use_topk else
                          _expected_reward_full(logits, by, tau))

            with torch.no_grad():
                oracle = by.max(dim=1).values
                cur = (F.softmax(logits / tau, dim=1) * by).sum(dim=1)
                gap = (oracle - cur).clamp(min=0.0)
                w = ((gap + 1e-6) ** cfg.hard_reweight_power)
                w = (w / (w.mean() + 1e-12)).clamp(max=cfg.hard_reweight_clip)

            loss = (w * per_sample).mean() \
                 + ce_w * F.cross_entropy(logits, by.argmax(dim=1)) \
                 - ent_bonus * _entropy(logits)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            router.eval()
            with torch.no_grad():
                val_er = sum(
                    (F.softmax(router(vx.to(device)) / tau, dim=1) * vy.to(device)).sum().item()
                    for vx, vy in val_loader
                ) / len(Xv)
                print(f"epoch {epoch:02d}  tau={tau:.3f}  val_ER={val_er:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": router.state_dict(), "config": cfg.__dict__,
                "num_experts": num_experts, "input_dim": X.shape[1]}, out_path)
    print(f"[save] {out_path}")

    best_single = Y.mean(dim=0).argmax().item()
    return {
        "num_experts": num_experts,
        "best_single_test_reward": float(Yte[:, best_single].mean()),
        "oracle_test_reward": float(Yte.max(dim=1).values.mean()),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("router.pt"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(train_router(args.checkpoint, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
