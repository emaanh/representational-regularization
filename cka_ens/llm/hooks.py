"""Forward-hook activation capture for HuggingFace decoder blocks."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Sequence

try:
    import torch
    from torch import nn
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[llm]'`.") from e


def select_anchor_layers(model: "nn.Module", every: int = 4) -> List[int]:
    """Return decoder-layer indices to hook (every K-th, 0-indexed)."""
    n = _count_decoder_layers(model)
    return list(range(every - 1, n, every))


def _count_decoder_layers(model: "nn.Module") -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise AttributeError(f"Unknown decoder structure on {type(model).__name__}.")


def _resolve_layer(model: "nn.Module", idx: int) -> "nn.Module":
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[idx]
    raise AttributeError("Unknown decoder structure.")


@contextmanager
def capture_layer_activations(
    model: "nn.Module",
    layer_indices: Sequence[int],
) -> Iterator[List["torch.Tensor"]]:
    """Context manager yielding a list of post-block hidden states."""
    captured: List["torch.Tensor"] = []
    handles = []

    def make_hook(_i: int):
        def hook(_m, _inp, outputs):
            captured.append(outputs[0] if isinstance(outputs, tuple) else outputs)
        return hook

    for i, idx in enumerate(layer_indices):
        handles.append(_resolve_layer(model, idx).register_forward_hook(make_hook(i)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
