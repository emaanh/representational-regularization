"""LLM sequential-diversification trainer + router (PyTorch + HF)."""
from .penalty import linear_cka_torch, sequential_cka_penalty

__all__ = ["linear_cka_torch", "sequential_cka_penalty"]
