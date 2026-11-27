# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Numerical regression tests for CUDA fused_logp and deterministic_logp_tile_stats."""

import math

import pytest
import torch


def _cuda_logp_available() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None:
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    except Exception:
        return False
    return _EXT_AVAILABLE and all(
        hasattr(_C, name)
        for name in (
            "fused_logp_forward_fp32",
            "fused_logp_forward_online_fp32",
            "fused_logp_backward",
            "deterministic_logp_tile_stats",
        )
    )


requires_cuda_logp = pytest.mark.skipif(
    not _cuda_logp_available(),
    reason="CUDA logprob extension symbols unavailable",
)


@requires_cuda_logp
def test_fused_logp_online_handles_neg_inf_logits_without_nan():
    """Rows containing multiple -inf entries must not evaluate exp(-inf - (-inf)) = NaN."""
    from rl_engine.backends.extension import _C

    logits = torch.tensor(
        [
            [float("-inf"), 0.0, float("-inf"), 1.0],
            [float("-inf"), float("-inf"), 2.0, float("-inf")],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    token_ids = torch.tensor([1, 2], device="cuda", dtype=torch.long)

    out_online = _C.fused_logp_forward_online_fp32(logits, token_ids)
    out_twopass = _C.fused_logp_forward_fp32(logits, token_ids)
    ref = torch.log_softmax(logits, dim=-1).gather(1, token_ids.unsqueeze(1)).squeeze(1)

    assert torch.isfinite(out_online).all(), f"Expected finite online logp, got {out_online}"
    torch.testing.assert_close(out_online, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_online, out_twopass, atol=1e-5, rtol=1e-5)


@requires_cuda_logp
def test_fused_logp_forward_and_backward_handle_logits_below_neg_1e20():
    """Finite FP32 rows with logits < -1e20 must not clamp max to -1e20 (+inf / NaN)."""
    from rl_engine.backends.extension import _C

    vocab = 8
    logits = torch.full((2, vocab), -1e25, device="cuda", dtype=torch.float32)
    token_ids = torch.tensor([0, 3], device="cuda", dtype=torch.long)
    grad_out = torch.ones((2,), device="cuda", dtype=torch.float32)

    out_fwd = _C.fused_logp_forward_fp32(logits, token_ids)
    out_online = _C.fused_logp_forward_online_fp32(logits, token_ids)
    grad_logits = _C.fused_logp_backward(logits, token_ids, grad_out)

    expected_logp = torch.full((2,), -math.log(vocab), device="cuda", dtype=torch.float32)
    assert torch.isfinite(out_fwd).all(), f"Forward returned non-finite: {out_fwd}"
    assert torch.isfinite(out_online).all(), f"Online forward returned non-finite: {out_online}"
    assert torch.isfinite(grad_logits).all(), f"Backward returned non-finite: {grad_logits}"
    torch.testing.assert_close(out_fwd, expected_logp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_online, expected_logp, atol=1e-5, rtol=1e-5)

    expected_grad = torch.full_like(logits, -1.0 / vocab)
    expected_grad[0, 0] += 1.0
    expected_grad[1, 3] += 1.0
    torch.testing.assert_close(grad_logits, expected_grad, atol=1e-5, rtol=1e-5)


@requires_cuda_logp
def test_deterministic_logp_tile_stats_padding_tile_returns_neg_inf():
    """Padding-only and all--inf tiles must report tile_max=-inf and tile_sum=0."""
    from rl_engine.backends.extension import _C

    torch.manual_seed(0)
    logits = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    tile_max, tile_sum = _C.deterministic_logp_tile_stats(logits, 0, 128, 2)

    assert torch.isneginf(tile_max[:, 1]).all(), f"Expected -inf padding tile_max, got {tile_max[:, 1]}"
    assert torch.equal(tile_sum[:, 1], torch.zeros(4, device="cuda", dtype=torch.float32))
