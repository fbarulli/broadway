"""GluonTS torch step adapter for the memory probe (DeepAR-class model family).

Implements the probe contract::

    step(batch_size, seq_len, device, dtype) -> parameter_count

It builds a real GluonTS torch model (SimpleFeedForward, the lightest in the
DeepAR/WaveNet family) and runs ONE forward + backward + optimizer step, so the
probe measures a genuine training footprint. ``gluonts``/``torch`` are imported
lazily: an absent package fails loud with the install hint instead of returning
a made-up number.
"""

from __future__ import annotations

_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
_HINT = 'the GluonTS probe adapter needs gluonts[torch]: pip install "gluonts[torch]"'


def step(batch_size: int, seq_len: int, device: str, dtype: str) -> int:
    """Run one training step of a GluonTS torch model; return its parameter count."""
    try:
        import torch
        from gluonts.torch.model.simple_feedforward.module import SimpleFeedForwardModel
    except ImportError as exc:
        raise ImportError(_HINT) from exc
    torch_dtype = getattr(torch, _DTYPES[dtype])
    model = SimpleFeedForwardModel(prediction_length=seq_len, context_length=seq_len).to(
        device=device, dtype=torch_dtype
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    past = torch.randn(batch_size, seq_len, device=device, dtype=torch_dtype)
    future = torch.randn(batch_size, seq_len, device=device, dtype=torch_dtype)
    observed = torch.ones(batch_size, seq_len, device=device, dtype=torch_dtype)
    loss = model.loss(
        past_target=past,
        future_target=future,
        future_observed_values=observed,
    ).mean()
    loss.backward()
    optimizer.step()
    return sum(parameter.numel() for parameter in model.parameters())
