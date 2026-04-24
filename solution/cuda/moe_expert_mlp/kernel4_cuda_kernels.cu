#include "kernel4_internal.cuh"

namespace kernel4_internal {

__global__ void fp8_gemm1_swiglu_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             T_e,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             seq_len)
{
    int tile_t  = blockIdx.x * BT;
    int tile_bn = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float A_smem[BT][BK];

    int local_col = tid;
    int up_col = tile_bn * BN + local_col;
    int gate_col = INTERMEDIATE_SIZE + tile_bn * BN + local_col;

    float acc_up[BT] = {};
    float acc_gate[BT] = {};

    for (int bk = 0; bk < NUM_HIDDEN_BLOCKS; ++bk) {
        int k_base = bk * BK;

        for (int t = 0; t < BT; ++t) {
            int tok = tile_t + t;
            if (tok < T_e && local_col < BK) {
                int orig_tok = load_cached(token_indices + tok);
                A_smem[t][local_col] = fp8_to_float(act[(size_t)orig_tok * HIDDEN_SIZE + k_base + local_col]);
            } else if (local_col < BK) {
                A_smem[t][local_col] = 0.f;
            }
        }


        __syncthreads();

        float w_scale_up = (up_col < GEMM1_OUT_SIZE)
            ? load_cached(W_scale + tile_bn * NUM_HIDDEN_BLOCKS + bk)
            : 0.f;
        float w_scale_gate = (gate_col < GEMM1_OUT_SIZE)
            ? load_cached(W_scale + (INTERMEDIATE_SIZE / BN + tile_bn) * NUM_HIDDEN_BLOCKS + bk)
            : 0.f;

        for (int t = 0; t < BT; ++t) {
            int tok = tile_t + t;
            int orig_tok = (tok < T_e) ? load_cached(token_indices + tok) : 0;
            float a_scale = (tok < T_e) ? load_cached(act_scale + bk * seq_len + orig_tok) : 0.f;

            float dot_up = 0.f;
            float dot_gate = 0.f;
            for (int k = 0; k < BK; ++k) {
                float w_up = (up_col < GEMM1_OUT_SIZE)
                    ? fp8_to_float(W[up_col * HIDDEN_SIZE + k_base + k])
                    : 0.f;
                float w_gate = (gate_col < GEMM1_OUT_SIZE)
                    ? fp8_to_float(W[gate_col * HIDDEN_SIZE + k_base + k])
                    : 0.f;
                dot_up += A_smem[t][k] * w_up;
                dot_gate += A_smem[t][k] * w_gate;
            }
            acc_up[t] += dot_up * a_scale * w_scale_up;
            acc_gate[t] += dot_gate * a_scale * w_scale_gate;
        }

        __syncthreads();
    }

    for (int t = 0; t < BT; ++t) {
        int tok = tile_t + t;
        if (tok >= T_e) break;

        float up = acc_up[t];
        float gate = acc_gate[t];
        float silu_up = up * (1.f / (1.f + expf(-up)));
        float result = gate * silu_up;

        out[tok * INTERMEDIATE_SIZE + up_col] = __float2bfloat16(result);
    }
}

__global__ void fp8_gemm1_swiglu_reference_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    const int*      __restrict__ local_expert_ids,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             total_tokens,
    int             seq_len)
{
    int tok = blockIdx.x;
    int out_col = threadIdx.x + blockIdx.y * blockDim.x;
    int row = threadIdx.x;

    if (tok >= total_tokens || out_col >= INTERMEDIATE_SIZE) return;

    int expert_id = load_cached(local_expert_ids + tok);

    int orig_tok = load_cached(token_indices + tok);
    const fp8_e4m3* W_e = W + (size_t)expert_id * GEMM1_OUT_SIZE * HIDDEN_SIZE;
    const float* Ws_e = W_scale + (size_t)expert_id * NUM_GEMM1_OUT_BLOCKS * NUM_HIDDEN_BLOCKS;

    float acc_up = 0.f;
    float acc_gate = 0.f;
    int bn_up = out_col / BLOCK_SIZE;
    int bn_gate = (INTERMEDIATE_SIZE + out_col) / BLOCK_SIZE;

    __shared__ uint8_t W_smem[256][BK];

    for (int bk = 0; bk < NUM_HIDDEN_BLOCKS; ++bk) {
        float a_scale = load_cached(act_scale + bk * seq_len + orig_tok);
        float ws_up = load_cached(Ws_e + bn_up * NUM_HIDDEN_BLOCKS + bk);
        float ws_gate = load_cached(Ws_e + bn_gate * NUM_HIDDEN_BLOCKS + bk);

        int k_base = bk * BLOCK_SIZE;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            W_smem[row][k] = load_cached(W_e + out_col * HIDDEN_SIZE + k_base + k);
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a = fp8_to_float(act[(size_t)orig_tok * HIDDEN_SIZE + k_base + k]) * a_scale;
            acc_up += a * fp8_to_float(W_smem[row][k]) * ws_up;
        }


        for (int k = 0; k < BLOCK_SIZE; ++k) {
            W_smem[row][k] = load_cached(W_e + (INTERMEDIATE_SIZE + out_col) * HIDDEN_SIZE + k_base + k);
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a = fp8_to_float(act[(size_t)orig_tok * HIDDEN_SIZE + k_base + k]) * a_scale;
            acc_gate += a * fp8_to_float(W_smem[row][k]) * ws_gate;
        }


        __syncthreads();
    }

    float silu_gate = acc_gate * (1.f / (1.f + expf(-acc_gate)));
    float result = acc_up * silu_gate;
    out[tok * INTERMEDIATE_SIZE + out_col] = __float2bfloat16(result);
}

__global__ void fp8_gemm1_swiglu_micro_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             T_e,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             seq_len)
{
    int tile_t = blockIdx.x * K4_MICRO_BT;
    int tile_bn = blockIdx.y;
    int tid = threadIdx.x;
    int out_col = tile_bn * K4_MICRO_BN + tid;

    __shared__ float A_smem[K4_MICRO_BT][BK];
    __shared__ uint8_t W_up_smem[K4_MICRO_BN][BK];
    __shared__ uint8_t W_gate_smem[K4_MICRO_BN][BK];

    float acc_up[K4_MICRO_BT] = {};
    float acc_gate[K4_MICRO_BT] = {};

    for (int bk = 0; bk < NUM_HIDDEN_BLOCKS; ++bk) {
        int k_base = bk * BK;

        for (int t = 0; t < K4_MICRO_BT; ++t) {
            int tok = tile_t + t;
            for (int k = tid; k < BK; k += K4_MICRO_BN) {
                if (tok < T_e) {
                    int orig_tok = load_cached(token_indices + tok);
                    A_smem[t][k] = fp8_to_float(
                        act[(size_t)orig_tok * HIDDEN_SIZE + k_base + k]);
                } else {
                    A_smem[t][k] = 0.f;
                }
            }
        }

        if (out_col < INTERMEDIATE_SIZE) {
            for (int k = 0; k < BK; ++k) {
                W_up_smem[tid][k] = load_cached(
                    W + (size_t)out_col * HIDDEN_SIZE + k_base + k);
                W_gate_smem[tid][k] = load_cached(
                    W + (size_t)(INTERMEDIATE_SIZE + out_col) * HIDDEN_SIZE + k_base + k);
            }
        }

        __syncthreads();

        if (out_col < INTERMEDIATE_SIZE) {
            float w_scale_up = load_cached(W_scale + tile_bn * NUM_HIDDEN_BLOCKS + bk);
            float w_scale_gate = load_cached(
                W_scale + (INTERMEDIATE_SIZE / K4_MICRO_BN + tile_bn) * NUM_HIDDEN_BLOCKS + bk);

            for (int t = 0; t < K4_MICRO_BT; ++t) {
                int tok = tile_t + t;
                if (tok >= T_e) {
                    break;
                }
                int orig_tok = load_cached(token_indices + tok);
                float a_scale = load_cached(act_scale + bk * seq_len + orig_tok);
                float dot_up = 0.f;
                float dot_gate = 0.f;
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    float a = A_smem[t][k];
                    dot_up += a * fp8_to_float(W_up_smem[tid][k]);
                    dot_gate += a * fp8_to_float(W_gate_smem[tid][k]);
                }
                acc_up[t] += dot_up * a_scale * w_scale_up;
                acc_gate[t] += dot_gate * a_scale * w_scale_gate;
            }
        }

        __syncthreads();
    }

    if (out_col >= INTERMEDIATE_SIZE) {
        return;
    }

    for (int t = 0; t < K4_MICRO_BT; ++t) {
        int tok = tile_t + t;
        if (tok >= T_e) {
            break;
        }
        float up = acc_up[t];
        float gate = acc_gate[t];
        float silu_gate = gate * (1.f / (1.f + expf(-gate)));
        out[(size_t)tok * INTERMEDIATE_SIZE + out_col] = __float2bfloat16(up * silu_gate);
    }
}

__global__ void fp8_gemm1_swiglu_small_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             T_e,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             seq_len)
{
    int tile_t  = blockIdx.x * K4_SMALL_BT;
    int tile_bn = blockIdx.y;
    int tid = threadIdx.x;
    int up_col = tile_bn * K4_SMALL_BN + tid;
    int gate_col = INTERMEDIATE_SIZE + tile_bn * K4_SMALL_BN + tid;

    __shared__ float A_smem[K4_SMALL_BT][BK];
    __shared__ uint8_t W_up_smem[K4_SMALL_BN][BK];
    __shared__ uint8_t W_gate_smem[K4_SMALL_BN][BK];

    float acc_up[K4_SMALL_BT] = {};
    float acc_gate[K4_SMALL_BT] = {};

    for (int bk = 0; bk < NUM_HIDDEN_BLOCKS; ++bk) {
        int k_base = bk * BK;

        for (int t = 0; t < K4_SMALL_BT; ++t) {
            int tok = tile_t + t;
            for (int k = tid; k < BK; k += K4_SMALL_BN) {
                if (tok < T_e) {
                    int orig_tok = load_cached(token_indices + tok);
                    A_smem[t][k] = fp8_to_float(
                        act[(size_t)orig_tok * HIDDEN_SIZE + k_base + k]);
                } else {
                    A_smem[t][k] = 0.f;
                }
            }
        }

        if (up_col < INTERMEDIATE_SIZE) {
            for (int k = 0; k < BK; ++k) {
                W_up_smem[tid][k] = load_cached(
                    W + (size_t)up_col * HIDDEN_SIZE + k_base + k);
                W_gate_smem[tid][k] = load_cached(
                    W + (size_t)gate_col * HIDDEN_SIZE + k_base + k);
            }
        }

        __syncthreads();

        if (up_col < INTERMEDIATE_SIZE) {
            float w_scale_up = load_cached(W_scale + tile_bn * NUM_HIDDEN_BLOCKS + bk);
            float w_scale_gate = load_cached(
                W_scale + (INTERMEDIATE_SIZE / K4_SMALL_BN + tile_bn) * NUM_HIDDEN_BLOCKS + bk);

            for (int t = 0; t < K4_SMALL_BT; ++t) {
                int tok = tile_t + t;
                if (tok >= T_e) {
                    break;
                }
                int orig_tok = load_cached(token_indices + tok);
                float a_scale = load_cached(act_scale + bk * seq_len + orig_tok);
                float dot_up = 0.f;
                float dot_gate = 0.f;
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    float a = A_smem[t][k];
                    dot_up += a * fp8_to_float(W_up_smem[tid][k]);
                    dot_gate += a * fp8_to_float(W_gate_smem[tid][k]);
                }
                acc_up[t] += dot_up * a_scale * w_scale_up;
                acc_gate[t] += dot_gate * a_scale * w_scale_gate;
            }
        }

        __syncthreads();
    }

    if (up_col >= INTERMEDIATE_SIZE) {
        return;
    }

    for (int t = 0; t < K4_SMALL_BT; ++t) {
        int tok = tile_t + t;
        if (tok >= T_e) {
            break;
        }
        float up = acc_up[t];
        float gate = acc_gate[t];
        float silu_gate = gate * (1.f / (1.f + expf(-gate)));
        out[(size_t)tok * INTERMEDIATE_SIZE + up_col] = __float2bfloat16(up * silu_gate);
    }
}


#if defined(K4_ENABLE_CUTLASS)
__global__ void dequant_activations_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             token_offset,
    int             token_count,
    int             seq_len,
    float*          __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = token_count * HIDDEN_SIZE;
    if (idx >= total) return;

    int local_tok = idx / HIDDEN_SIZE;
    int k = idx % HIDDEN_SIZE;
    int global_tok = token_offset + local_tok;
    int orig_tok = token_indices[global_tok];
    int bk = k / BLOCK_SIZE;
    float scale = act_scale[bk * seq_len + orig_tok];
    out[idx] = fp8_to_float(act[(size_t)global_tok * HIDDEN_SIZE + k]) * scale;
}

__global__ void dequant_gemm1_weight_half_kernel(
    const fp8_e4m3* __restrict__ weights,
    const float*    __restrict__ scales,
    int             row_offset,
    float*          __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = INTERMEDIATE_SIZE * HIDDEN_SIZE;
    if (idx >= total) return;

    int row = idx / HIDDEN_SIZE;
    int k = idx % HIDDEN_SIZE;
    int global_row = row_offset + row;
    int bn = global_row / BLOCK_SIZE;
    int bk = k / BLOCK_SIZE;
    float scale = scales[bn * NUM_HIDDEN_BLOCKS + bk];
    out[idx] = fp8_to_float(weights[(size_t)global_row * HIDDEN_SIZE + k]) * scale;
}

__global__ void swiglu_pack_kernel(
    const float* __restrict__ up,
    const float* __restrict__ gate,
    int          token_offset,
    int          token_count,
    __nv_bfloat16* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = token_count * INTERMEDIATE_SIZE;
    if (idx >= total) return;

    int local_tok = idx / INTERMEDIATE_SIZE;
    float up_val = up[idx];
    float gate_val = gate[idx];
    float silu_up = up_val * (1.f / (1.f + expf(-up_val)));
    out[(size_t)(token_offset + local_tok) * INTERMEDIATE_SIZE + (idx % INTERMEDIATE_SIZE)] =
        __float2bfloat16(gate_val * silu_up);
}
#endif

}  // namespace kernel4_internal
