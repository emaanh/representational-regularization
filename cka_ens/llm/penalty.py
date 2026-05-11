"""PyTorch linear-CKA penalty."""
from __future__ import annotations

from typing import Sequence

try:
    import torch
    from torch import Tensor
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e


def _flatten(X: "Tensor") -> "Tensor":
    return X if X.dim() <= 2 else X.reshape(X.shape[0], -1)


def _centering(K: "Tensor") -> "Tensor":
    return K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()


def linear_cka_torch(X: "Tensor", Y: "Tensor", eps: float = 1e-7) -> "Tensor":
    """Differentiable linear CKA between batch-aligned tensors."""
    X = _flatten(X.float())
    Y = _flatten(Y.float())
    Gx = _centering(X @ X.t())
    Gy = _centering(Y @ Y.t())
    return (Gx * Gy).sum() / (torch.sqrt((Gx * Gx).sum()) * torch.sqrt((Gy * Gy).sum()) + eps)


def sequential_cka_penalty(
    current_layer_acts: Sequence["Tensor"],
    prev_layer_acts_per_member: Sequence[Sequence["Tensor"]],
) -> "Tensor":
    """Mean over prior members of mean-over-layers CKA."""
    if not prev_layer_acts_per_member:
        return current_layer_acts[0].new_tensor(0.0)
    total = current_layer_acts[0].new_tensor(0.0)
    for prev in prev_layer_acts_per_member:
        per_layer = [linear_cka_torch(c, p) for c, p in zip(current_layer_acts, prev)]
        total = total + torch.stack(per_layer).mean()
    return total / float(len(prev_layer_acts_per_member))
