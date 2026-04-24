#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstddef>

// ── Compile-time constants matching the spec ─────────────────────────────────
namespace moe_spec {
    constexpr int NUM_EXPERTS          = 256;
    constexpr int NUM_LOCAL_EXPERTS    = 32;
    constexpr int HIDDEN_SIZE          = 7168;
    constexpr int INTERMEDIATE_SIZE    = 2048;
    constexpr int GEMM1_OUT_SIZE       = 4096;   // = 2 * INTERMEDIATE_SIZE (W1||W3 fused)
    constexpr int NUM_HIDDEN_BLOCKS    = 56;     // 7168 / 128
    constexpr int NUM_GEMM1_OUT_BLOCKS = 32;     // 4096 / 128
    constexpr int NUM_INTER_BLOCKS     = 16;     // 2048 / 128
    constexpr int BLOCK_SIZE           = 128;    // universal quantization block size
    constexpr int TOP_K                = 8;      // DeepSeek-V3 top-8 routing
}

// Stored as raw bytes and decoded in the kernels.
using fp8_e4m3 = uint8_t;

enum class Kernel4Backend : int {
    Auto = 0,
    Fallback = 1,
    Tiled = 2,
    Cutlass = 3,
};

struct Kernel4Problem {
    const float*          routing_logits;       // [seq_len, NUM_EXPERTS]
    const __nv_bfloat16*  routing_bias;         // [NUM_EXPERTS]
    int                   seq_len;

    const fp8_e4m3*       hidden_states;        // [total_dispatched_tokens, HIDDEN_SIZE]
    const float*          hidden_states_scale;  // [NUM_HIDDEN_BLOCKS, seq_len]

    const fp8_e4m3*       gemm1_weights;        // [NUM_LOCAL_EXPERTS, GEMM1_OUT_SIZE, HIDDEN_SIZE]
    const float*          gemm1_weights_scale;  // [NUM_LOCAL_EXPERTS, NUM_GEMM1_OUT_BLOCKS, NUM_HIDDEN_BLOCKS]
    const fp8_e4m3*       gemm2_weights;        // [NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE]
    const float*          gemm2_weights_scale;  // [NUM_LOCAL_EXPERTS, NUM_HIDDEN_BLOCKS, NUM_INTER_BLOCKS]

    int                   local_expert_offset;    // this rank owns experts [offset, offset+32)
    float                 routed_scaling_factor;  // multiply final weighted sum
    const int*            expert_token_offsets;   // [NUM_LOCAL_EXPERTS + 1], device ptr
    const int*            host_expert_token_offsets = nullptr; // [NUM_LOCAL_EXPERTS + 1], optional host mirror
    int                   total_dispatched_tokens = -1;   // optional precomputed total, <0 means query from device
    const int*            token_indices;          // [total_dispatched_tokens], device ptr
    const int*            local_expert_ids;       // [total_dispatched_tokens], device ptr
    const float*          token_expert_weights;   // [total_dispatched_tokens], device ptr

    __nv_bfloat16*        output;                 // [seq_len, HIDDEN_SIZE]

    Kernel4Backend        backend;
    cudaStream_t          stream;
};

struct Kernel4Workspace {
    __nv_bfloat16* gemm1_output;     // [total_dispatched_tokens, INTERMEDIATE_SIZE]
    float*         gemm2_output;     // [total_dispatched_tokens, HIDDEN_SIZE]
    float*         output_accum;     // [seq_len, HIDDEN_SIZE]

    void*          storage;
    size_t         storage_bytes;

    void*          cutlass_workspace;
    size_t         cutlass_workspace_bytes;
};

// Query bytes for a single contiguous workspace allocation.
size_t k4_query_workspace(int seq_len,
                          int total_dispatched_tokens,
                          size_t cutlass_workspace_bytes = 0);

// Slice a caller-owned workspace allocation into typed buffers.
Kernel4Workspace k4_bind_workspace(void* storage,
                                   size_t storage_bytes,
                                   int seq_len,
                                   int total_dispatched_tokens,
                                   size_t cutlass_workspace_bytes = 0);

// True when this binary can execute the CUTLASS backend on the active device.
bool k4_cutlass_available();

cudaError_t k4_launch(const Kernel4Problem& problem,
                      const Kernel4Workspace& workspace);

cudaError_t k4_launch_gemm1(const Kernel4Problem& problem,
                            const Kernel4Workspace& workspace);

// Slow host reference used by tests.
void k4_reference_cpu(
    const float*   hidden_states_f32,      // [total_dispatched_tokens, HIDDEN_SIZE]
    const float*   hidden_states_scale,    // [NUM_HIDDEN_BLOCKS, seq_len]
    const int*     token_indices,          // [total_dispatched_tokens]
    const int*     expert_token_offsets,   // [NUM_LOCAL_EXPERTS + 1]
    const float*   gemm1_weights_f32,      // [NUM_LOCAL_EXPERTS, GEMM1_OUT_SIZE, HIDDEN_SIZE]
    const float*   gemm1_weights_scale,    // [NUM_LOCAL_EXPERTS, NUM_GEMM1_OUT_BLOCKS, NUM_HIDDEN_BLOCKS]
    const float*   gemm2_weights_f32,      // [NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE]
    const float*   gemm2_weights_scale,    // [NUM_LOCAL_EXPERTS, NUM_HIDDEN_BLOCKS, NUM_INTER_BLOCKS]
    const float*   token_expert_weights,   // [total_dispatched_tokens]
    float          routed_scaling_factor,
    int            seq_len,
    int            total_dispatched_tokens,
    float*         output_f32              // [seq_len, HIDDEN_SIZE]
);
