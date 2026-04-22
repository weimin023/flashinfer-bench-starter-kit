#include <cuda_runtime.h>

template<int E_LOCAL, int TOP_K>
__global__ void moe_reindex_kernel(
    const int* __restrict__ token_expert_indices,
    const float* __restrict__ token_expert_weights,
    const int* __restrict__ token_expert_slots,
    const int* __restrict__ expert_token_offsets,
    int* __restrict__ token_indices,
    float* __restrict__ merged_token_weights,
    int seq_len,
    int local_expert_offset) {

    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * TOP_K;
    if (linear >= total) return;

    int token = linear / TOP_K;
    int global_expert = token_expert_indices[linear];
    int local_expert = global_expert - local_expert_offset;

    if (local_expert < 0 || local_expert >= E_LOCAL) return;

    int slot = token_expert_slots[linear];
    int dest = expert_token_offsets[local_expert] + slot;
    token_indices[dest] = token;
    merged_token_weights[dest] = token_expert_weights[linear];
}

void launch_moe_reindex(
    const int* token_expert_indices,
    const float* token_expert_weights,
    const int* token_expert_slots,
    const int* expert_token_offsets,
    int* token_indices,
    float* merged_token_weights,
    int seq_len,
    int local_expert_offset,
    cudaStream_t stream = 0) {

    constexpr int E_LOCAL = 32;
    constexpr int TOP_K = 8;
    int total = seq_len * TOP_K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    moe_reindex_kernel<E_LOCAL, TOP_K><<<blocks, threads, 0, stream>>>(
        token_expert_indices,
        token_expert_weights,
        token_expert_slots,
        expert_token_offsets,
        token_indices,
        merged_token_weights,
        seq_len,
        local_expert_offset);
}
