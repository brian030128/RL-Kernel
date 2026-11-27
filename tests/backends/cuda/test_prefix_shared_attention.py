# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regression and correctness tests for CUDA prefix_shared_attention."""

import math

import pytest
import torch


def _cuda_prefix_attn_available() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None:
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    except Exception:
        return False
    return _EXT_AVAILABLE and hasattr(_C, "prefix_shared_attention")


requires_cuda_prefix_attn = pytest.mark.skipif(
    not _cuda_prefix_attn_available(),
    reason="CUDA _C.prefix_shared_attention extension is unavailable",
)


def _reference_prefix_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: [bs, G, len_q, dim], k/v: [bs, len_kv, dim]
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().unsqueeze(1).transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float().unsqueeze(1))
    return out.to(q.dtype)


@requires_cuda_prefix_attn
@pytest.mark.parametrize(
    "bs,G,len_q,len_kv",
    [
        (2, 2, 64, 64),
        (2, 2, 17, 50),
        (2, 3, 70, 100),
        (1, 4, 65, 130),
    ],
)
def test_prefix_shared_attention_matches_reference_including_non_multiple_of_64(
    bs, G, len_q, len_kv
):
    """Non-64-aligned len_q and len_kv must not read/write OOB or leak padding into softmax."""
    from rl_engine.backends.extension import _C

    torch.manual_seed(100 + len_q + len_kv)
    dim = 128
    q = torch.randn((bs, G, len_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((bs, len_kv, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((bs, len_kv, dim), device="cuda", dtype=torch.bfloat16)

    out = _C.prefix_shared_attention(q, k, v)
    torch.cuda.synchronize()
    ref = _reference_prefix_attn(q, k, v)

    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=2e-2)


@requires_cuda_prefix_attn
def test_prefix_shared_attention_cuda_graph_capture_uses_current_stream():
    """Kernel must launch on at::cuda::getCurrentCUDAStream(), not legacy stream 0."""
    from rl_engine.backends.extension import _C

    torch.manual_seed(7)
    q = torch.randn((1, 2, 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 64, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 64, 128), device="cuda", dtype=torch.bfloat16)

    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        warm = _C.prefix_shared_attention(q, k, v)
    side_stream.synchronize()
    torch.testing.assert_close(
        warm.float(), _reference_prefix_attn(q, k, v).float(), atol=3e-2, rtol=2e-2
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        out = _C.prefix_shared_attention(q, k, v)

    new_q = torch.randn_like(q)
    q.copy_(new_q)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out.float(), _reference_prefix_attn(new_q, k, v).float(), atol=3e-2, rtol=2e-2
    )


@requires_cuda_prefix_attn
def test_prefix_shared_attention_invalid_dim_raises_runtime_error():
    """Unsupported head dim must raise RuntimeError via TORCH_CHECK, not call exit(1)."""
    from rl_engine.backends.extension import _C

    q = torch.randn((1, 1, 64, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 64, 64), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 64, 64), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="dim=128"):
        _C.prefix_shared_attention(q, k, v)
