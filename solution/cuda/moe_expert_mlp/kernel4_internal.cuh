#pragma once

#include "kernel4.cuh"
#include "kernel6_internal.cuh"

#include <cstdio>
#include <cmath>

using namespace moe_spec;

#define CUDA_CHECK(x)                                                  \
    do { cudaError_t _e=(x);                                           \
        if(_e!=cudaSuccess){                                           \
            fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,        \
                    cudaGetErrorString(_e)); return _e; }              \
    } while(0)

namespace kernel4_internal {

constexpr int BT = 4;
constexpr int BN = 128;
constexpr int BK = 128;
constexpr int K4_MICRO_BT = 2;
constexpr int K4_MICRO_BN = 64;
constexpr int K4_SMALL_BT = 4;
constexpr int K4_SMALL_BN = 64;
constexpr int K4_GROUPED_SMALL_BT = 1;
constexpr int K4_GROUPED_SMALL_BN = 64;
constexpr size_t kWorkspaceAlign = kernel6_internal::kWorkspaceAlign;

using kernel6_internal::load_cached;

inline size_t align_up(size_t value, size_t alignment = kWorkspaceAlign) {
    return (value + alignment - 1) / alignment * alignment;
}

inline size_t gemm1_output_bytes(int total_dispatched_tokens) {
    return (size_t)total_dispatched_tokens * INTERMEDIATE_SIZE * sizeof(__nv_bfloat16);
}

__device__ __forceinline__ float fp8_to_float(uint8_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __nv_fp8_e4m3 x;
    x.__x = v;
    return (float)x;
#else
    if (v == 0x7F || v == 0xFF) return 0.f;
    int sign = (v >> 7) & 1;
    int exp_bits = (v >> 3) & 0xF;
    int man_bits = v & 0x7;
    float mantissa = (exp_bits == 0) ? (man_bits / 8.0f) : (1.0f + man_bits / 8.0f);
    float scale = (exp_bits == 0) ? ldexpf(1.0f, -6) : ldexpf(1.0f, exp_bits - 7);
    float result = mantissa * scale;
    return sign ? -result : result;
#endif
}

__global__ void fp8_gemm1_swiglu_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             T_e,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             seq_len);

__global__ void fp8_gemm1_swiglu_reference_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    const int*      __restrict__ local_expert_ids,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             total_tokens,
    int             seq_len);

__global__ void fp8_gemm1_swiglu_micro_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             T_e,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             seq_len);

__global__ void fp8_gemm1_swiglu_small_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             T_e,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             seq_len);

__global__ void fp8_gemm1_swiglu_grouped_small_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    const int*      __restrict__ local_expert_ids,
    const fp8_e4m3* __restrict__ W,
    const float*    __restrict__ W_scale,
    __nv_bfloat16*  __restrict__ out,
    int             total_tokens,
    int             seq_len);

#if defined(K4_ENABLE_CUTLASS)
__global__ void dequant_activations_kernel(
    const fp8_e4m3* __restrict__ act,
    const float*    __restrict__ act_scale,
    const int*      __restrict__ token_indices,
    int             token_offset,
    int             token_count,
    int             seq_len,
    float*          __restrict__ out);

__global__ void dequant_gemm1_weight_half_kernel(
    const fp8_e4m3* __restrict__ weights,
    const float*    __restrict__ scales,
    int             row_offset,
    float*          __restrict__ out);

__global__ void swiglu_pack_kernel(
    const float* __restrict__ up,
    const float* __restrict__ gate,
    int          token_offset,
    int          token_count,
    __nv_bfloat16* __restrict__ out);

size_t cutlass_aux_bytes(int total_dispatched_tokens);
cudaError_t launch_cutlass_backend(const Kernel4Problem& p,
                                   const Kernel4Workspace& workspace,
                                   int total_tokens,
                                   bool gemm1_only = false);
#endif

cudaError_t launch_fallback_backend(const Kernel4Problem& p,
                                    const Kernel4Workspace& workspace,
                                    int total_tokens,
                                    bool gemm1_only = false);

cudaError_t launch_tiled_backend(const Kernel4Problem& p,
                                 const Kernel4Workspace& workspace,
                                 int total_tokens,
                                 bool gemm1_only = false);

}  // namespace kernel4_internal
