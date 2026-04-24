#include "kernel6_internal.cuh"

namespace kernel6_internal {

__global__ void fp8_gemm2_tiled_project_and_combine_kernel(
    const __nv_bfloat16* __restrict__ inter,
    const int*           __restrict__ token_indices,
    const float*         __restrict__ routing_w,
    const fp8_e4m3*      __restrict__ W2,
    const float*         __restrict__ W2_scale,
    float                routed_scaling_factor,
    float*               __restrict__ output_accum,
    int                  token_offset,
    int                  token_count,
    int                  seq_len)
{
    int tile_tok = blockIdx.x * K6_TILE_T;
    int tile_h = blockIdx.y;
    int local_h = threadIdx.x;
    int out_col = tile_h * K6_TILE_H + local_h;

    __shared__ float inter_smem[K6_TILE_T][BLOCK_SIZE];
    __shared__ uint8_t w_smem[K6_TILE_H][BLOCK_SIZE];

    float acc[K6_TILE_T] = {};

    for (int ib = 0; ib < NUM_INTER_BLOCKS; ++ib) {
        int k_base = ib * BLOCK_SIZE;

        for (int t = 0; t < K6_TILE_T; ++t) {
            int tok = tile_tok + t;
            if (tok < token_count && local_h < BLOCK_SIZE) {
                inter_smem[t][local_h] = __bfloat162float(
                    inter[(size_t)(token_offset + tok) * INTERMEDIATE_SIZE + k_base + local_h]);
            } else if (local_h < BLOCK_SIZE) {
                inter_smem[t][local_h] = 0.f;
            }
        }

        if (out_col < HIDDEN_SIZE) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                w_smem[threadIdx.x][k] = load_cached(
                    W2 + (size_t)out_col * INTERMEDIATE_SIZE + k_base + k);
            }
        }

        __syncthreads();

        if (out_col < HIDDEN_SIZE) {
            float tile_scale = load_cached(
                W2_scale + (out_col / BLOCK_SIZE) * NUM_INTER_BLOCKS + ib);
            for (int t = 0; t < K6_TILE_T; ++t) {
                int tok = tile_tok + t;
                if (tok >= token_count) {
                    break;
                }
                float dot = 0.f;
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    dot += inter_smem[t][k] * fp8_to_float(w_smem[threadIdx.x][k]);
                }
                acc[t] += dot * tile_scale;
            }
        }

        __syncthreads();
    }

    if (out_col >= HIDDEN_SIZE) {
        return;
    }

    for (int t = 0; t < K6_TILE_T; ++t) {
        int tok = tile_tok + t;
        if (tok >= token_count) {
            break;
        }
        int orig_tok = load_cached(token_indices + token_offset + tok);
        if (orig_tok < 0 || orig_tok >= seq_len) {
            continue;
        }
        float rw = load_cached(routing_w + token_offset + tok) * routed_scaling_factor;
        atomicAdd(output_accum + (size_t)orig_tok * HIDDEN_SIZE + out_col, acc[t] * rw);
    }
}

__global__ void fp8_gemm2_micro_project_and_combine_kernel(
    const __nv_bfloat16* __restrict__ inter,
    const int*           __restrict__ token_indices,
    const float*         __restrict__ routing_w,
    const fp8_e4m3*      __restrict__ W2,
    const float*         __restrict__ W2_scale,
    float                routed_scaling_factor,
    float*               __restrict__ output_accum,
    int                  token_offset,
    int                  token_count,
    int                  seq_len)
{
    int tile_tok = blockIdx.x * K6_MICRO_TILE_T;
    int tile_h = blockIdx.y;
    int lane = threadIdx.x;
    int out_col = tile_h * K6_MICRO_TILE_H + lane;

    __shared__ float inter_smem[K6_MICRO_TILE_T][BLOCK_SIZE];
    __shared__ uint8_t w_smem[K6_MICRO_TILE_H][BLOCK_SIZE];

    float acc[K6_MICRO_TILE_T] = {};

    for (int ib = 0; ib < NUM_INTER_BLOCKS; ++ib) {
        int k_base = ib * BLOCK_SIZE;

        for (int t = 0; t < K6_MICRO_TILE_T; ++t) {
            int tok = tile_tok + t;
            for (int k = lane; k < BLOCK_SIZE; k += K6_MICRO_TILE_H) {
                inter_smem[t][k] = (tok < token_count)
                    ? __bfloat162float(inter[(size_t)(token_offset + tok) * INTERMEDIATE_SIZE + k_base + k])
                    : 0.f;
            }
        }

        if (out_col < HIDDEN_SIZE) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                w_smem[lane][k] = load_cached(
                    W2 + (size_t)out_col * INTERMEDIATE_SIZE + k_base + k);
            }
        }

        __syncthreads();

        if (out_col < HIDDEN_SIZE) {
            float tile_scale = load_cached(
                W2_scale + (out_col / BLOCK_SIZE) * NUM_INTER_BLOCKS + ib);
            for (int t = 0; t < K6_MICRO_TILE_T; ++t) {
                int tok = tile_tok + t;
                if (tok >= token_count) {
                    break;
                }
                float dot = 0.f;
                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    dot += inter_smem[t][k] * fp8_to_float(w_smem[lane][k]);
                }
                acc[t] += dot * tile_scale;
            }
        }

        __syncthreads();
    }

    if (out_col >= HIDDEN_SIZE) {
        return;
    }

    for (int t = 0; t < K6_MICRO_TILE_T; ++t) {
        int tok = tile_tok + t;
        if (tok >= token_count) {
            break;
        }
        int orig_tok = load_cached(token_indices + token_offset + tok);
        if (orig_tok < 0 || orig_tok >= seq_len) {
            continue;
        }
        float rw = load_cached(routing_w + token_offset + tok) * routed_scaling_factor;
        atomicAdd(output_accum + (size_t)orig_tok * HIDDEN_SIZE + out_col, acc[t] * rw);
    }
}

__global__ void fp8_gemm2_project_and_combine_kernel(
    const __nv_bfloat16* __restrict__ inter,
    const int*           __restrict__ local_expert_ids,
    const int*           __restrict__ token_indices,
    const float*         __restrict__ routing_w,
    const fp8_e4m3*      __restrict__ W2,
    const float*         __restrict__ W2_scale,
    float*               __restrict__ output_accum,
    int                  total_tokens)
{
    int tok = blockIdx.x;
    int out_col = threadIdx.x + blockIdx.y * blockDim.x;
    if (tok >= total_tokens || out_col >= HIDDEN_SIZE) return;

    int expert_id = load_cached(local_expert_ids + tok);

    const __nv_bfloat16* inter_tok = inter + (size_t)tok * INTERMEDIATE_SIZE;
    const fp8_e4m3* W_e = W2 + (size_t)expert_id * HIDDEN_SIZE * INTERMEDIATE_SIZE;
    const float* Ws_e = W2_scale + (size_t)expert_id * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;

    int hb = out_col / BLOCK_SIZE;
    float acc = 0.f;
    for (int ib = 0; ib < NUM_INTER_BLOCKS; ++ib) {
        float tile_scale = load_cached(Ws_e + hb * NUM_INTER_BLOCKS + ib);
        int k_base = ib * BLOCK_SIZE;
        float dot = 0.f;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a = __bfloat162float(inter_tok[k_base + k]);
            float w = fp8_to_float(load_cached(W_e + out_col * INTERMEDIATE_SIZE + k_base + k));
            dot += a * w;
        }
        acc += dot * tile_scale;
    }


    int orig_tok = load_cached(token_indices + tok);
    float rw = load_cached(routing_w + tok);
    atomicAdd(output_accum + (size_t)orig_tok * HIDDEN_SIZE + out_col, acc * rw);
}

__global__ void combine_projected_kernel(
    const float*   __restrict__ projected,
    const int*     __restrict__ token_indices,
    const float*   __restrict__ routing_w,
    float          routed_scaling_factor,
    float*         __restrict__ output_accum,
    int            total_tokens,
    int            seq_len)
{
    int tok = blockIdx.x;
    int out_col = threadIdx.x + blockIdx.y * blockDim.x;
    if (tok >= total_tokens || out_col >= HIDDEN_SIZE) return;

    int orig_tok = load_cached(token_indices + tok);
    if (orig_tok < 0 || orig_tok >= seq_len) return;

    float rw = load_cached(routing_w + tok);
    float value = projected[(size_t)tok * HIDDEN_SIZE + out_col];
    atomicAdd(output_accum + (size_t)orig_tok * HIDDEN_SIZE + out_col, value * rw);
}

__global__ void f32_to_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

__global__ void bf16_rows_to_f32_kernel(
    const __nv_bfloat16* __restrict__ input,
    int                  token_offset,
    int                  token_count,
    int                  row_stride,
    float*               __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = token_count * row_stride;
    if (idx >= total) return;

    int local_tok = idx / row_stride;
    int col = idx % row_stride;
    output[idx] = __bfloat162float(input[(size_t)(token_offset + local_tok) * row_stride + col]);
}

#if defined(K4_ENABLE_CUTLASS)
__global__ void dequant_gemm2_weight_kernel(
    const fp8_e4m3* __restrict__ weights,
    const float*    __restrict__ scales,
    float*          __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
    if (idx >= total) return;

    int row = idx / INTERMEDIATE_SIZE;
    int k = idx % INTERMEDIATE_SIZE;
    int hb = row / BLOCK_SIZE;
    int ib = k / BLOCK_SIZE;
    float scale = scales[hb * NUM_INTER_BLOCKS + ib];
    out[idx] = fp8_to_float(weights[(size_t)row * INTERMEDIATE_SIZE + k]) * scale;
}
#endif

}  // namespace kernel6_internal
