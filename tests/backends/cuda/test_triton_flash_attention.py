# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regression tests for Triton FlashAttention bounds, non-contiguous strides, and BF16 backward."""

import pytest
import torch
import torch.nn.functional as F

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton FlashAttention requires CUDA and Triton",
)


def _run_attn_fwd_bwd(fn, q, k, v, *, causal: bool):
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    out = fn(q_, k_, v_, causal=causal)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    return out.detach(), q_.grad.detach(), k_.grad.detach(), v_.grad.detach(), grad_out


@requires_triton_cuda
@pytest.mark.parametrize("seq_len", [50, 70, 100])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_flash_attention_non_multiple_of_64_and_bf16_backward(seq_len, causal, dtype):
    """Non-64-aligned seq_len and BF16 backward must match SDPA without OOB or atomic_add crash."""
    from rl_engine.backends.shared.triton.attention.triton_attn import triton_flash_attention

    torch.manual_seed(seq_len + int(causal))
    q = torch.randn((2, 4, seq_len, 64), device="cuda", dtype=dtype)
    k = torch.randn((2, 4, seq_len, 64), device="cuda", dtype=dtype)
    v = torch.randn((2, 4, seq_len, 64), device="cuda", dtype=dtype)

    q_tri = q.clone().requires_grad_(True)
    k_tri = k.clone().requires_grad_(True)
    v_tri = v.clone().requires_grad_(True)
    out_tri = triton_flash_attention(q_tri, k_tri, v_tri, causal=causal)
    grad_out = torch.randn_like(out_tri)
    out_tri.backward(grad_out)

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=causal)
    out_ref.backward(grad_out.float())

    atol = 3e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(out_tri.float(), out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q_tri.grad.float(), q_ref.grad, atol=atol * 2, rtol=rtol * 2)
    torch.testing.assert_close(k_tri.grad.float(), k_ref.grad, atol=atol * 2, rtol=rtol * 2)
    torch.testing.assert_close(v_tri.grad.float(), v_ref.grad, atol=atol * 2, rtol=rtol * 2)


@requires_triton_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_triton_flash_attention_non_contiguous_bshd_transpose_layout(causal):
    """[B, S, H, D].transpose(1, 2) inputs must use each tensor's (z, h) strides cleanly."""
    from rl_engine.backends.shared.triton.attention.triton_attn import triton_flash_attention

    torch.manual_seed(99 + int(causal))
    q_base = torch.randn((2, 70, 4, 64), device="cuda", dtype=torch.float16)
    k_base = torch.randn((2, 70, 4, 64), device="cuda", dtype=torch.float16)
    v_base = torch.randn((2, 70, 4, 64), device="cuda", dtype=torch.float16)

    q_tri = q_base.transpose(1, 2).detach().requires_grad_(True)
    k_tri = k_base.transpose(1, 2).detach().requires_grad_(True)
    v_tri = v_base.transpose(1, 2).detach().requires_grad_(True)
    assert not q_tri.is_contiguous()

    out_tri = triton_flash_attention(q_tri, k_tri, v_tri, causal=causal)
    grad_out = torch.randn_like(out_tri)
    out_tri.backward(grad_out)

    q_ref = q_base.transpose(1, 2).float().detach().requires_grad_(True)
    k_ref = k_base.transpose(1, 2).float().detach().requires_grad_(True)
    v_ref = v_base.transpose(1, 2).float().detach().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=causal)
    out_ref.backward(grad_out.float())

    torch.testing.assert_close(out_tri.float(), out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q_tri.grad.float(), q_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_tri.grad.float(), k_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v_tri.grad.float(), v_ref.grad, atol=2e-2, rtol=2e-2)
