"""Small CIFAR-10 CNN with four hooked layers."""
from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[cifar]'`.") from e


LAYER_NAMES = ("c1", "c2", "c3", "d1")


class TinyCNN(nn.Module):
    """3-conv + 1-dense classifier."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.c3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, return_acts: bool = False):
        c1 = F.relu(self.c1(x))
        c2 = F.relu(self.c2(self.pool(c1)))
        c3 = F.relu(self.c3(self.pool(c2)))
        h = c3.flatten(1)
        d1 = F.relu(self.fc1(h))
        logits = self.fc2(d1)
        if return_acts:
            return logits, [c1, c2, c3, d1]
        return logits


def get_model_name(n: int) -> str:
    """Excel-style names (A..Z, AA, AB, ...)."""
    name = ""
    while n >= 0:
        name = chr(ord("A") + n % 26) + name
        n = n // 26 - 1
    return name
