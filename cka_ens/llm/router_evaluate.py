"""Evaluate a trained router top-1 or top-K, broken down by benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e

from .router_train import ResidualRouter


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--router", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--topk", type=int, default=1)
    return p.parse_args(argv)


def _benchmark_breakdown(probs, rewards, benchmarks, topk):
    """Per-benchmark router/best-single/oracle accuracy."""
    by_bench: dict = {}
    for i, b in enumerate(benchmarks):
        by_bench.setdefault(b, []).append(i)
    out = {}
    for b, idxs in by_bench.items():
        idxs_t = torch.tensor(idxs)
        p = probs[idxs_t]; r = rewards[idxs_t]
        vals, idx = p.topk(topk, dim=1)
        vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
        per_sample = (vals * r.gather(1, idx)).sum(dim=1)
        out[b] = {
            "n": len(idxs),
            "router_acc": float((per_sample > 0).float().mean()),
            "best_single": float(r.mean(dim=0).max()),
            "oracle": float(r.max(dim=1).values.mean()),
        }
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.router)
    router = ResidualRouter(blob["input_dim"], blob["num_experts"]).to(device)
    router.load_state_dict(blob["state_dict"])
    router.eval()

    data = torch.load(args.data)
    X = (data["X"].to(device) if isinstance(data["X"], torch.Tensor)
         else torch.tensor(data["X"], device=device, dtype=torch.float32))
    Y = data["Y_rewards"].float() if "Y_rewards" in data else data["Y"].float()

    with torch.no_grad():
        probs = F.softmax(router(X), dim=1).cpu()

    per_bench = _benchmark_breakdown(probs, Y, data["benchmarks"], args.topk)
    print(f"{'Benchmark':<16} {'n':>5} {'BestSingle':>12} {'Router':>10} {'Oracle':>10}")
    for b, m in per_bench.items():
        print(f"{b:<16} {m['n']:>5d} {m['best_single']*100:>11.1f}% "
              f"{m['router_acc']*100:>9.1f}% {m['oracle']*100:>9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
