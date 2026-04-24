#include "kernel6_internal.cuh"

namespace kernel6_internal {

cudaError_t launch_fallback_gemm2_combine(const Gemm2Problem& p,
                                          const Gemm2Workspace& workspace,
                                          int total_tokens) {
    K6_CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        output_accum_bytes(p.seq_len),
        p.stream));

    dim3 block(256);
    dim3 gemm2_grid(total_tokens, (HIDDEN_SIZE + block.x - 1) / block.x);
    fp8_gemm2_project_and_combine_kernel<<<gemm2_grid, block, 0, p.stream>>>(
        p.hidden_states,
        p.local_expert_ids,
        p.token_indices,
        p.token_expert_weights,
        p.gemm2_weights,
        p.gemm2_weights_scale,
        workspace.output_accum,
        total_tokens
    );
    K6_CUDA_CHECK(cudaGetLastError());

    int total_output_elems = p.seq_len * HIDDEN_SIZE;
    dim3 pack_grid((total_output_elems + block.x - 1) / block.x);
    f32_to_bf16_kernel<<<pack_grid, block, 0, p.stream>>>(
        workspace.output_accum,
        p.output,
        total_output_elems
    );
    K6_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t launch_tiled_gemm2_combine(const Gemm2Problem& p,
                                       const Gemm2Workspace& workspace,
                                       int total_tokens) {
    K6_CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        output_accum_bytes(p.seq_len),
        p.stream));

    int host_offsets[NUM_LOCAL_EXPERTS + 1];
    K6_CUDA_CHECK(cudaMemcpy(
        host_offsets,
        p.expert_token_offsets,
        sizeof(host_offsets),
        cudaMemcpyDeviceToHost));

    dim3 tiled_block(K6_TILE_H);
    dim3 micro_block(K6_MICRO_TILE_H);
    for (int expert = 0; expert < NUM_LOCAL_EXPERTS; ++expert) {
        int begin = host_offsets[expert];
        int end = host_offsets[expert + 1];
        int token_count = end - begin;
        if (token_count <= 0) {
            continue;
        }

        const __nv_bfloat16* inter_e = p.hidden_states;
        const fp8_e4m3* w_e =
            p.gemm2_weights + (size_t)expert * HIDDEN_SIZE * INTERMEDIATE_SIZE;
        const float* ws_e =
            p.gemm2_weights_scale + (size_t)expert * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;

        if (token_count <= 4) {
            dim3 micro_grid(
                (token_count + K6_MICRO_TILE_T - 1) / K6_MICRO_TILE_T,
                HIDDEN_SIZE / K6_MICRO_TILE_H);
            fp8_gemm2_micro_project_and_combine_kernel<<<micro_grid, micro_block, 0, p.stream>>>(
                inter_e,
                p.token_indices,
                p.token_expert_weights,
                w_e,
                ws_e,
                p.routed_scaling_factor,
                workspace.output_accum,
                begin,
                token_count,
                p.seq_len);
        } else {
            dim3 tiled_grid(
                (token_count + K6_TILE_T - 1) / K6_TILE_T,
                HIDDEN_SIZE / K6_TILE_H);
            fp8_gemm2_tiled_project_and_combine_kernel<<<tiled_grid, tiled_block, 0, p.stream>>>(
                inter_e,
                p.token_indices,
                p.token_expert_weights,
                w_e,
                ws_e,
                p.routed_scaling_factor,
                workspace.output_accum,
                begin,
                token_count,
                p.seq_len);
        }
        K6_CUDA_CHECK(cudaGetLastError());
    }

    int total_output_elems = p.seq_len * HIDDEN_SIZE;
    dim3 pack_grid((total_output_elems + 255) / 256);
    f32_to_bf16_kernel<<<pack_grid, 256, 0, p.stream>>>(
        workspace.output_accum,
        p.output,
        total_output_elems
    );
    K6_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

}  // namespace kernel6_internal
