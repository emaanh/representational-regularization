"""CIFAR-10 dataloaders via torchvision."""
from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError as e:
    raise ImportError("Install with `uv pip install -e '.[cifar]'`.") from e


def get_loaders(batch_size: int = 64, data_root: str = "./cifar10_data"):
    """Return (train_loader, test_loader, sample_x)."""
    tx = transforms.Compose([transforms.ToTensor()])
    Path(data_root).mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(data_root, train=True, download=True, transform=tx)
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=tx)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    sample_x = torch.stack([test[i][0] for i in range(batch_size)])
    return train_loader, test_loader, sample_x
