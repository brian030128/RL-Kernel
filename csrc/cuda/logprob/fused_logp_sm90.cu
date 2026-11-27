// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include "../utils/tma_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <math_constants.h>
#include <torch/extension.h>

namespace {

// A single TMA box dimension is capped at 256 elements on Hopper.
#define TILE_V 4096
#define TMA_BOX 256
static constexpr int LOADS_PER_TILE = TILE_V / TMA_BOX;

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int NUM_WARPS, bool USE_TMA>
__global__ void fused_logp_online_tma_kernel(
    const __grid_constant__ CUtensorMap logits_tmap,
    const int* __restrict__ labels,
    const nv_bfloat16* __restrict__ logits_gmem,
    float* __restrict__ output_logp,
    int batch_size,
    int vocab_size)
{
    static_assert(NUM_WARPS > 1, "fused_logp_online_tma_kernel requires at least 2 warps");
    constexpr int NUM_CONSUMER_WARPS = NUM_WARPS - 1;
    constexpr int NUM_CONSUMERS = NUM_CONSUMER_WARPS * 32;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int row_idx = blockIdx.x;

    extern __shared__ __align__(1024) char smem[];
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    nv_bfloat16* smem_logits = reinterpret_cast<nv_bfloat16*>(smem);

    const uint32_t tma_mbar_addr = smem_addr + static_cast<uint32_t>(TILE_V * sizeof(nv_bfloat16));
    const uint32_t mma_mbar_addr = tma_mbar_addr + 8u;

    if constexpr (USE_TMA) {
        if (warp_id == 0 && lane_id == 0) {
            mbarrier_init(tma_mbar_addr, 1);
            mbarrier_init(mma_mbar_addr, NUM_CONSUMERS);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();
    }

    const int v_aligned = USE_TMA ? ((vocab_size / TMA_BOX) * TMA_BOX) : 0;
    const int num_tiles = USE_TMA ? ((v_aligned + TILE_V - 1) / TILE_V) : 0;
    int phase = 0;

    if (warp_id == 0) {
        if constexpr (USE_TMA) {
            if (lane_id == 0) {
                for (int step = 0; step < num_tiles; ++step) {
                    const int col_offset = step * TILE_V;
                    const int current_tile_size = min(TILE_V, v_aligned - col_offset);

                    if (step > 0) {
                        mbarrier_wait(mma_mbar_addr, phase ^ 1);
                    }

                    for (int j = 0; j < LOADS_PER_TILE; ++j) {
                        const int col = col_offset + j * TMA_BOX;
                        if (col < v_aligned) {
                            tma_2d_g2s(
                                smem_addr + static_cast<uint32_t>(j * TMA_BOX * sizeof(nv_bfloat16)),
                                &logits_tmap, col, row_idx, tma_mbar_addr);
                        }
                    }
                    mbarrier_arrive_expect_tx(
                        tma_mbar_addr, static_cast<int>(current_tile_size * sizeof(nv_bfloat16)));
                    phase ^= 1;
                }
            }
        }
    } else {
        const int consumer_warp = warp_id - 1;
        const int consumer_tid = consumer_warp * 32 + lane_id;

        __shared__ float s_warp_buf[NUM_CONSUMER_WARPS];
        __shared__ float s_tile_max;

        float row_max = -CUDART_INF_F;
        float row_sum = 0.0f;

        if constexpr (USE_TMA) {
            for (int step = 0; step < num_tiles; ++step) {
                const int col_offset = step * TILE_V;
                const int current_tile_size = min(TILE_V, v_aligned - col_offset);

                mbarrier_wait(tma_mbar_addr, phase);

                float tile_max = -CUDART_INF_F;
                for (int i = consumer_tid; i < current_tile_size; i += NUM_CONSUMERS) {
                    const float val = __bfloat162float(smem_logits[i]);
                    tile_max = fmaxf(tile_max, val);
                }
                tile_max = warp_reduce_max(tile_max);
                if (lane_id == 0) {
                    s_warp_buf[consumer_warp] = tile_max;
                }
                asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");

                if (consumer_tid == 0) {
                    float m = s_warp_buf[0];
#pragma unroll
                    for (int w = 1; w < NUM_CONSUMER_WARPS; ++w) {
                        m = fmaxf(m, s_warp_buf[w]);
                    }
                    s_tile_max = m;
                }
                asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");

                if (s_tile_max != -CUDART_INF_F) {
                    float tile_sum = 0.0f;
                    for (int i = consumer_tid; i < current_tile_size; i += NUM_CONSUMERS) {
                        const float val = __bfloat162float(smem_logits[i]);
                        tile_sum += expf(val - s_tile_max);
                    }
                    tile_sum = warp_reduce_sum(tile_sum);
                    if (lane_id == 0) {
                        s_warp_buf[consumer_warp] = tile_sum;
                    }
                    asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");

                    if (consumer_tid == 0) {
                        float block_tile_sum = 0.0f;
#pragma unroll
                        for (int w = 0; w < NUM_CONSUMER_WARPS; ++w) {
                            block_tile_sum += s_warp_buf[w];
                        }
                        if (row_max == -CUDART_INF_F) {
                            row_max = s_tile_max;
                            row_sum = block_tile_sum;
                        } else {
                            const float new_max = fmaxf(row_max, s_tile_max);
                            row_sum = row_sum * expf(row_max - new_max) +
                                      block_tile_sum * expf(s_tile_max - new_max);
                            row_max = new_max;
                        }
                    }
                    asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");
                }

                mbarrier_arrive(mma_mbar_addr);
                phase ^= 1;
            }
        }

        // Tail [v_aligned, vocab_size), or full row when USE_TMA is false.
        const int tail = vocab_size - v_aligned;
        if (tail > 0) {
            const int64_t base = static_cast<int64_t>(row_idx) * vocab_size + v_aligned;
            float tail_max = -CUDART_INF_F;
            for (int i = consumer_tid; i < tail; i += NUM_CONSUMERS) {
                const float val = __bfloat162float(logits_gmem[base + i]);
                tail_max = fmaxf(tail_max, val);
            }
            tail_max = warp_reduce_max(tail_max);
            if (lane_id == 0) {
                s_warp_buf[consumer_warp] = tail_max;
            }
            asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");

            if (consumer_tid == 0) {
                float m = s_warp_buf[0];
#pragma unroll
                for (int w = 1; w < NUM_CONSUMER_WARPS; ++w) {
                    m = fmaxf(m, s_warp_buf[w]);
                }
                s_tile_max = m;
            }
            asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");

            if (s_tile_max != -CUDART_INF_F) {
                float tail_sum = 0.0f;
                for (int i = consumer_tid; i < tail; i += NUM_CONSUMERS) {
                    const float val = __bfloat162float(logits_gmem[base + i]);
                    tail_sum += expf(val - s_tile_max);
                }
                tail_sum = warp_reduce_sum(tail_sum);
                if (lane_id == 0) {
                    s_warp_buf[consumer_warp] = tail_sum;
                }
                asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");

                if (consumer_tid == 0) {
                    float block_tail_sum = 0.0f;
#pragma unroll
                    for (int w = 0; w < NUM_CONSUMER_WARPS; ++w) {
                        block_tail_sum += s_warp_buf[w];
                    }
                    if (row_max == -CUDART_INF_F) {
                        row_max = s_tile_max;
                        row_sum = block_tail_sum;
                    } else {
                        const float new_max = fmaxf(row_max, s_tile_max);
                        row_sum = row_sum * expf(row_max - new_max) +
                                  block_tail_sum * expf(s_tile_max - new_max);
                        row_max = new_max;
                    }
                }
                asm volatile("bar.sync 1, %0;" :: "n"(NUM_CONSUMERS) : "memory");
            }
        }

        if (consumer_tid == 0) {
            const int label_idx = labels[row_idx];
            const float label_val = __bfloat162float(
                logits_gmem[static_cast<int64_t>(row_idx) * vocab_size + label_idx]);
            const float lse =
                (row_max == -CUDART_INF_F) ? -CUDART_INF_F : (row_max + logf(row_sum));
            output_logp[row_idx] = label_val - lse;
        }
    }
}

}  // namespace

torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(logits.device());
    TORCH_CHECK(labels.is_cuda() && labels.device() == logits.device(),
                "labels must be a CUDA tensor on the same device as logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [B, V]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be bfloat16");

    const int B = static_cast<int>(logits.size(0));
    const int V = static_cast<int>(logits.size(1));
    TORCH_CHECK(labels.numel() == B, "labels must have one entry per row");

    auto output = torch::empty({B}, logits.options().dtype(torch::kFloat));
    if (B == 0) {
        return output;
    }

    auto labels_i = labels.to(torch::kInt32).contiguous();
    const nv_bfloat16* logits_ptr =
        reinterpret_cast<const nv_bfloat16*>(logits.data_ptr<at::BFloat16>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Hopper TMA requires 16-byte-aligned base pointer and row stride, plus V >= TMA_BOX.
    const bool use_tma =
        V >= TMA_BOX &&
        (reinterpret_cast<std::uintptr_t>(logits_ptr) % 16 == 0) &&
        ((static_cast<int64_t>(V) * sizeof(nv_bfloat16)) % 16 == 0);

    if (use_tma) {
        CUtensorMap logits_tmap;
        init_tensor_map<nv_bfloat16>(&logits_tmap, logits_ptr, B, V, 1, TMA_BOX);
        const int smem_size = (TILE_V * sizeof(nv_bfloat16)) + 16;
        fused_logp_online_tma_kernel<4, true><<<B, 128, smem_size, stream>>>(
            logits_tmap, labels_i.data_ptr<int>(), logits_ptr,
            output.data_ptr<float>(), B, V);
    } else {
        CUtensorMap unused_tmap{};
        fused_logp_online_tma_kernel<4, false><<<B, 128, 0, stream>>>(
            unused_tmap, labels_i.data_ptr<int>(), logits_ptr,
            output.data_ptr<float>(), B, V);
    }
    return output;
}
