# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regression and correctness tests for the SM90 TMA fused_logp_sm90 CUDA kernel."""

import pytest
import torch


def _sm90_fused_logp_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    except Exception:
        return False
    return (
        _EXT_AVAILABLE
        and hasattr(_C, "fused_logp_sm90")
        and torch.cuda.get_device_capability()[0] == 9
    )


requires_sm90_fused_logp = pytest.mark.skipif(
    not _sm90_fused_logp_available(),
    reason=(
        "fused_logp_sm90 kernel not compiled "
        "(needs KERNEL_ALIGN_FORCE_SM90=1 on an SM90/Hopper device)."
    ),
)


def _reference_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.long().unsqueeze(1)).squeeze(1)


@requires_sm90_fused_logp
@pytest.mark.parametrize(
    "vocab_size",
    [
        128,     # V < TMA_BOX (256) -> non-TMA global fallback
        256,     # exact single TMA_BOX
        4096,    # exact TILE_V (16 x TMA_BOX)
        4100,    # 8-byte stride (not 16-byte TMA aligned) -> global fallback
        4104,    # 16-byte aligned TMA + global tail [4096, 4104)
        151936,  # Qwen3-8B vocab size (TMA tiles + global tail)
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_fused_logp_sm90_matches_reference(batch_size, vocab_size):
    """TMA box <= 256, global tail, non-TMA fallback, and consumer sync must match reference."""
    from rl_engine.backends.extension import _C

    torch.manual_seed(42 + batch_size + vocab_size)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (batch_size,), device="cuda", dtype=torch.int32)

    out = _C.fused_logp_sm90(logits, labels)
    ref = _reference_logp(logits, labels)

    assert out.shape == (batch_size,)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=2e-3, rtol=1e-3)


@requires_sm90_fused_logp
def test_fused_logp_sm90_batch_invariance():
    """Row 0 output must be bitwise identical between batch=1 and batch=16."""
    from rl_engine.backends.extension import _C

    torch.manual_seed(7)
    vocab_size = 8192
    row = torch.randn(1, vocab_size, device="cuda", dtype=torch.bfloat16)
    label = torch.tensor([1234], device="cuda", dtype=torch.int32)
    single = _C.fused_logp_sm90(row, label)

    batch = torch.randn(16, vocab_size, device="cuda", dtype=torch.bfloat16)
    batch_labels = torch.randint(0, vocab_size, (16,), device="cuda", dtype=torch.int32)
    batch[0] = row[0]
    batch_labels[0] = label[0]
    batched = _C.fused_logp_sm90(batch, batch_labels)

    assert torch.equal(single[0], batched[0])


@requires_sm90_fused_logp
def test_fused_logp_sm90_all_neg_inf_tiles_and_tail_no_nan():
    """All--inf TMA tiles or global tails must not produce -inf - (-inf) = NaN."""
    from rl_engine.backends.extension import _C

    torch.manual_seed(11)
    # Two 4096-element TMA tiles + 104-element 16-byte-aligned tail.
    logits = torch.randn(3, 8296, device="cuda", dtype=torch.bfloat16)
    logits[0, :4096] = float("-inf")   # leading tile all -inf
    logits[1, 4096:] = float("-inf")   # second tile + tail all -inf
    logits[2, :8192] = float("-inf")   # all TMA tiles -inf, only tail finite
    labels = torch.tensor([5000, 123, 8200], device="cuda", dtype=torch.int32)

    out = _C.fused_logp_sm90(logits, labels)
    ref = _reference_logp(logits, labels)

    assert torch.isfinite(out).all(), f"Expected finite logp for masked tiles, got {out}"
    assert torch.allclose(out, ref, atol=2e-3, rtol=1e-3)


@requires_sm90_fused_logp
def test_fused_logp_sm90_cuda_graph_and_non_default_stream():
    """Kernel must launch on at::cuda::getCurrentCUDAStream() so CUDA graphs capture cleanly."""
    from rl_engine.backends.extension import _C

    torch.manual_seed(19)
    logits = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 4096, (8,), device="cuda", dtype=torch.int32)

    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        warm = _C.fused_logp_sm90(logits, labels)
    side_stream.synchronize()
    assert torch.allclose(warm, _reference_logp(logits, labels), atol=2e-3)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        out = _C.fused_logp_sm90(logits, labels)

    new_logits = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    logits.copy_(new_logits)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.allclose(out, _reference_logp(new_logits, labels), atol=2e-3)


@requires_sm90_fused_logp
def test_fused_logp_sm90_empty_batch():
    """Empty batch B=0 must return an empty float32 tensor without calling TMA setup."""
    from rl_engine.backends.extension import _C

    logits = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)
    labels = torch.empty(0, device="cuda", dtype=torch.int32)

    out = _C.fused_logp_sm90(logits, labels)
    torch.cuda.synchronize()
    assert out.shape == (0,)
    assert out.dtype == torch.float32
