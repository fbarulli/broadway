"""Tiny step adapter for the memory probe (CPU or GPU).

This is the reference implementation of the adapter contract:

    step(batch_size, seq_len, device, dtype) -> parameter_count

It runs one real forward + backward + optimizer step so the probe can measure
the footprint. Replace it with an adapter around the production trainer on the
GPU host (see gpu_training/README.md).
"""

from __future__ import annotations

import torch
from torch import nn


def step(batch_size: int, seq_len: int, device: str, dtype: str) -> int:
    """Run one training step and return the model's parameter count."""
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    model = nn.Sequential(
        nn.Linear(seq_len, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device=device, dtype=torch_dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    features = torch.randn(batch_size, seq_len, device=device, dtype=torch_dtype)
    target = torch.randn(batch_size, 1, device=device, dtype=torch_dtype)
    loss = ((model(features) - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    return sum(parameter.numel() for parameter in model.parameters())
