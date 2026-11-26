"""TimesFM 2.5 torch step adapter for the memory probe.

Implements the probe contract::

    step(batch_size, seq_len, device, dtype) -> parameter_count

TimesFM 2.5 ships a randomly-initialised torch module
(``TimesFM_2p5_200M_torch_module``); the probe builds it without downloading a
checkpoint (that is the zero-shot path) and runs ONE forward + backward +
optimizer step over random patches, so the measured footprint is the 200M
backbone's real training footprint. ``timesfm``/``torch`` import lazily and an
absent package fails loud with the install hint.
"""

from __future__ import annotations

_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
_HINT = 'the TimesFM probe adapter needs timesfm[torch]: pip install "timesfm[torch]"'


def step(batch_size: int, seq_len: int, device: str, dtype: str) -> int:
    """Run one training step of the TimesFM 2.5 backbone; return its parameter count."""
    try:
        import torch
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    except ImportError as exc:
        raise ImportError(_HINT) from exc
    torch_dtype = getattr(torch, _DTYPES[dtype])
    model = TimesFM_2p5_200M_torch_module().to(device=device, dtype=torch_dtype)
    # TimesFM consumes 32-point patches; keep at least one patch for tiny seq_len.
    patches = max(1, seq_len // model.p)
    inputs = torch.randn(batch_size, patches, model.p, device=device, dtype=torch_dtype)
    masks = torch.zeros(batch_size, patches, model.p, device=device, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    (_, _, point, quantile_spread), _ = model(inputs, masks)
    loss = point.mean() + quantile_spread.mean()
    loss.backward()
    optimizer.step()
    return sum(parameter.numel() for parameter in model.parameters())
