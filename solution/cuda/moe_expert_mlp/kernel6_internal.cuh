#pragma once

#include "kernel6.cuh"

#include <cstdio>
#include <cmath>

#define K6_CUDA_CHECK(x)                                               \
    do { cudaError_t _e=(x);                                           \
        if(_e!=cudaSuccess){                                           \
            fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,        \
                    cudaGetErrorString(_e)); return _e; }              \
    } while(0)

namespace kernel6_internal {

using namespace moe_spec;

constexpr size_t kWorkspaceAlign = 256;
constexpr int K6_TILE_T = 4;
constexpr int K6_TILE_H = 64;
constexpr int K6_MICRO_TILE_T = 2;
constexpr int K6_MICRO_TILE_H = 32;

inline size_t align_up(size_t value, size_t alignment = kWorkspaceAlign) {
    return (value + alignment - 1) / alignment * alignment;
}

inline size_t gemm2_output_bytes(int total_dispatched_tokens) {
    return (size_t)total_dispatched_tokens * HIDDEN_SIZE * sizeof(float);
}

inline size_t output_accum_bytes(int seq_len) {
    return (size_t)seq_len * HIDDEN_SIZE * sizeof(float);
}

__device__ __forceinline__ float load_cached(const float* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ int load_cached(const int* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ uint8_t load_cached(const uint8_t* ptr) {
    return __ldg(ptr);
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

struct Gemm2Problem {
    const __nv_bfloat16* hidden_states;
    const fp8_e4m3*      gemm2_weights;
    const float*         gemm2_weights_scale;
    const int*           expert_token_offsets;
    const int*           token_indices;
    const int*           local_expert_ids;
    const float*         token_expert_weights;
    float                routed_scaling_factor;
    int                  seq_len;
    cudaStream_t         stream;
    __nv_bfloat16*       output;
};

struct Gemm2Workspace {
    float*  gemm2_output;
    float*  output_accum;
    void*   cutlass_workspace;
    size_t  cutlass_workspace_bytes;
};

__global__ void fp8_gemm2_project_and_combine_kernel(
    const __nv_bfloat16* __restrict__ inter,
    const int*           __restrict__ local_expert_ids,
    const int*           __restrict__ token_indices,
    const float*         __restrict__ routing_w,
    const fp8_e4m3*      __restrict__ W2,
    const float*         __restrict__ W2_scale,
    float*               __restrict__ output_accum,
    int                  total_tokens);

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
    int                  seq_len);

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
    int                  seq_len);

__global__ void combine_projected_kernel(
    const float*   __restrict__ projected,
    const int*     __restrict__ token_indices,
    const float*   __restrict__ routing_w,
    float          routed_scaling_factor,
    float*         __restrict__ output_accum,
    int            total_tokens,
    int            seq_len);

__global__ void f32_to_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int n);

__global__ void bf16_rows_to_f32_kernel(
    const __nv_bfloat16* __restrict__ input,
    int                  token_offset,
    int                  token_count,
    int                  row_stride,
    float*               __restrict__ output);

#if defined(K4_ENABLE_CUTLASS)
__global__ void dequant_gemm2_weight_kernel(
    const fp8_e4m3* __restrict__ weights,
    const float*    __restrict__ scales,
    float*          __restrict__ out);

size_t cutlass_aux_bytes(int total_dispatched_tokens);
bool current_device_is_sm86_or_better();
cudaError_t launch_cutlass_gemm2_combine(const Gemm2Problem& p,
                                         const Gemm2Workspace& workspace,
                                         int total_tokens);
#endif

cudaError_t launch_fallback_gemm2_combine(const Gemm2Problem& p,
                                          const Gemm2Workspace& workspace,
                                          int total_tokens);

cudaError_t launch_tiled_gemm2_combine(const Gemm2Problem& p,
                                       const Gemm2Workspace& workspace,
                                       int total_tokens);

}  // namespace kernel6_internal
