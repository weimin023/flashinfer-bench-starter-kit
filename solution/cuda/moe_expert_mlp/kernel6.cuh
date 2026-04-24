#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstddef>

#include "kernel4.cuh"

enum class Kernel6Backend : int {
    Auto = 0,
    Fallback = 1,
    Tiled = 2,
    Cutlass = 3,
};

struct Kernel6Problem {
    const __nv_bfloat16*  hidden_states;        // [total_dispatched_tokens, INTERMEDIATE_SIZE]
    int                   seq_len;

    const fp8_e4m3*       gemm2_weights;        // [NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE]
    const float*          gemm2_weights_scale;  // [NUM_LOCAL_EXPERTS, NUM_HIDDEN_BLOCKS, NUM_INTER_BLOCKS]

    int                   local_expert_offset;
    float                 routed_scaling_factor;
    const int*            expert_token_offsets;   // [NUM_LOCAL_EXPERTS + 1], device ptr
    int                   total_dispatched_tokens = -1; // optional precomputed total, <0 means query from device
    const int*            token_indices;          // [total_dispatched_tokens], device ptr
    const int*            local_expert_ids;       // [total_dispatched_tokens], device ptr
    const float*          token_expert_weights;   // [total_dispatched_tokens], device ptr

    __nv_bfloat16*        output;                 // [seq_len, HIDDEN_SIZE]

    Kernel6Backend        backend;
    cudaStream_t          stream;
};

struct Kernel6Workspace {
    float*         gemm2_output;     // [total_dispatched_tokens, HIDDEN_SIZE]
    float*         output_accum;     // [seq_len, HIDDEN_SIZE]

    void*          storage;
    size_t         storage_bytes;

    void*          cutlass_workspace;
    size_t         cutlass_workspace_bytes;
};

size_t k6_query_workspace(int seq_len,
                          int total_dispatched_tokens,
                          size_t cutlass_workspace_bytes = 0);

Kernel6Workspace k6_bind_workspace(void* storage,
                                   size_t storage_bytes,
                                   int seq_len,
                                   int total_dispatched_tokens,
                                   size_t cutlass_workspace_bytes = 0);

bool k6_cutlass_available();

cudaError_t k6_launch(const Kernel6Problem& problem,
                      const Kernel6Workspace& workspace);

void k6_reference_cpu(
    const __nv_bfloat16* hidden_states_bf16,   // [total_dispatched_tokens, INTERMEDIATE_SIZE]
    const int*           token_indices,        // [total_dispatched_tokens]
    const int*           expert_token_offsets, // [NUM_LOCAL_EXPERTS + 1]
    const float*         gemm2_weights_f32,    // [NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE]
    const float*         gemm2_weights_scale,  // [NUM_LOCAL_EXPERTS, NUM_HIDDEN_BLOCKS, NUM_INTER_BLOCKS]
    const float*         token_expert_weights, // [total_dispatched_tokens]
    float                routed_scaling_factor,
    int                  seq_len,
    int                  total_dispatched_tokens,
    float*               output_f32            // [seq_len, HIDDEN_SIZE]
);
