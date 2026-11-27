import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < N_CTX

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Determine loop bounds for K and V
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        n_cols = start_n + offs_n[None, :]
        valid_mask = n_cols < N_CTX
        if IS_CAUSAL:
            valid_mask = valid_mask & (offs_m[:, None] >= n_cols)
        qk = tl.where(valid_mask, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        finite = m_i_new != float("-inf")
        alpha = tl.where(finite, tl.exp(m_i - m_i_new), 1.0)
        beta = tl.where(finite[:, None], tl.exp(qk - m_i_new[:, None]), 0.0)
        l_i_new = alpha * l_i + tl.sum(beta, 1)

        l_safe = tl.where(l_i_new > 0.0, l_i_new, 1.0)
        p_scale = beta / l_safe[:, None]
        acc_scale = (l_i / l_safe) * alpha

        acc = acc * acc_scale[:, None]
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        p = p_scale.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    acc = acc.to(Out.dtype.element_ty)
    tl.store(O_block_ptr, acc, boundary_check=(0, 1))

    off_zh = off_hz.to(tl.int64) * N_CTX
    l_ptrs = L + off_zh + offs_m
    m_ptrs = M + off_zh + offs_m
    tl.store(l_ptrs, l_i, mask=m_mask)
    tl.store(m_ptrs, m_i, mask=m_mask)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)

    o_offset = off_z * stride_oz + off_h * stride_oh
    do_offset = off_z * stride_doz + off_h * stride_doh

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(tl.program_id(0) * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_dom, stride_don),
        offsets=(tl.program_id(0) * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )

    o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    off_zh = off_hz.to(tl.int64) * N_CTX
    tl.store(Delta + off_zh + off_m, delta, mask=off_m < N_CTX)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
    Delta,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvk,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    do_offset = off_z * stride_doz + off_h * stride_doh
    dq_offset = off_z * stride_dqz + off_h * stride_dqh
    dk_offset = off_z * stride_dkz + off_h * stride_dkh
    dv_offset = off_z * stride_dvz + off_h * stride_dvh

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    lo = start_n * BLOCK_N if IS_CAUSAL else 0
    hi = N_CTX

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dom, stride_don),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    n_valid = offs_n < N_CTX
    off_zh = off_hz.to(tl.int64) * N_CTX

    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_valid = offs_m < N_CTX

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_i = tl.load(M + off_zh + offs_m, mask=m_valid, other=0.0)
        l_i = tl.load(L + off_zh + offs_m, mask=m_valid, other=1.0)
        delta = tl.load(Delta + off_zh + offs_m, mask=m_valid, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        valid_mask = m_valid[:, None] & n_valid[None, :]
        if IS_CAUSAL:
            valid_mask = valid_mask & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid_mask, qk, float("-inf"))

        p = tl.where(valid_mask, tl.exp(qk - m_i[:, None]) / l_i[:, None], 0.0)

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        ds = p * (dp - delta[:, None]) * sm_scale

        dq_val = tl.dot(ds.to(q.dtype), k)
        dq_ptrs = (
            DQ
            + dq_offset
            + (offs_m[:, None].to(tl.int64) * stride_dqm + offs_d[None, :].to(tl.int64) * stride_dqk)
        )
        tl.atomic_add(dq_ptrs, dq_val, mask=m_valid[:, None])

        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))

    DK_block_ptr = tl.make_block_ptr(
        base=DK + dk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dkn, stride_dkk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + dv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_dvn, stride_dvk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(DK_block_ptr, dk.to(DK.dtype.element_ty), boundary_check=(0, 1))
    tl.store(DV_block_ptr, dv.to(DV.dtype.element_ty), boundary_check=(0, 1))


class _TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # [batch, num_heads, seq_len, head_dim]
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk == Lv
        assert Lq in {16, 32, 64, 128, 256}

        ctx.sm_scale = sm_scale
        ctx.causal = causal

        batch, heads, seq_len, head_dim = q.shape

        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        M = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        L = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64 if head_dim > 64 else 128

        grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads, 1)

        device_module = torch.get_device_module(q.device)
        with device_module.device(q.device):
            _fwd_kernel[grid](
                q,
                k,
                v,
                sm_scale,
                L,
                M,
                out,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                batch,
                heads,
                seq_len,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=head_dim,
                IS_CAUSAL=causal,
                num_warps=4,
                num_stages=2,
            )

        ctx.save_for_backward(q, k, v, out, L, M)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, L, M = ctx.saved_tensors

        do = do.contiguous()
        # Accumulate dq in fp32 so atomic_add works on bf16/fp16 without precision loss.
        dq_fp32 = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
        dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)

        batch, heads, seq_len, head_dim = q.shape
        delta = torch.empty_like(L)

        BLOCK_M = 64
        BLOCK_N = 64

        device_module = torch.get_device_module(q.device)
        with device_module.device(q.device):
            grid_prep = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
            _bwd_preprocess[grid_prep](
                out,
                do,
                delta,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                batch,
                heads,
                seq_len,
                BLOCK_M=BLOCK_M,
                D_HEAD=head_dim,
            )

            grid_bwd = (triton.cdiv(seq_len, BLOCK_N), batch * heads, 1)
            _bwd_kernel[grid_bwd](
                q,
                k,
                v,
                ctx.sm_scale,
                out,
                do,
                dq_fp32,
                dk,
                dv,
                L,
                M,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                dq_fp32.stride(0),
                dq_fp32.stride(1),
                dq_fp32.stride(2),
                dq_fp32.stride(3),
                dk.stride(0),
                dk.stride(1),
                dk.stride(2),
                dk.stride(3),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                dv.stride(3),
                batch,
                heads,
                seq_len,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=head_dim,
                IS_CAUSAL=ctx.causal,
                num_warps=4,
                num_stages=1,
            )

        return dq_fp32.to(q.dtype), dk, dv, None, None


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
):
    """
    Universal backup Triton FlashAttention (support Forward / Backward)

    Args:
        q, k, v: Tensors of shape [batch, num_heads, seq_len, head_dim].
        causal: Whether to turn on causal masking.
        sm_scale: Softmax scaling factor, default to 1.0 / sqrt(head_dim).
    Returns:
        Attention Output tensor of shape [batch, num_heads, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    return _TritonAttention.apply(q, k, v, causal, sm_scale)
