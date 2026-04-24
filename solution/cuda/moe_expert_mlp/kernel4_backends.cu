#include "kernel4_internal.cuh"

namespace kernel4_internal {

cudaError_t launch_fallback_backend(const Kernel4Problem& p,
                                    const Kernel4Workspace& workspace,
                                    int total_tokens,
                                    bool gemm1_only) {
    dim3 block(256);
    dim3 gemm1_grid(total_tokens, (INTERMEDIATE_SIZE + block.x - 1) / block.x);
    fp8_gemm1_swiglu_reference_kernel<<<gemm1_grid, block, 0, p.stream>>>(
        p.hidden_states,
        p.hidden_states_scale,
        p.token_indices,
        p.local_expert_ids,
        p.gemm1_weights,
        p.gemm1_weights_scale,
        workspace.gemm1_output,
        total_tokens,
        p.seq_len
    );
    CUDA_CHECK(cudaGetLastError());
    if (gemm1_only) {
        return cudaSuccess;
    }

    CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        kernel6_internal::output_accum_bytes(p.seq_len),
        p.stream));

    kernel6_internal::Gemm2Problem shared_problem{};
    shared_problem.hidden_states = workspace.gemm1_output;
    shared_problem.gemm2_weights = p.gemm2_weights;
    shared_problem.gemm2_weights_scale = p.gemm2_weights_scale;
    shared_problem.expert_token_offsets = p.expert_token_offsets;
    shared_problem.token_indices = p.token_indices;
    shared_problem.local_expert_ids = p.local_expert_ids;
    shared_problem.token_expert_weights = p.token_expert_weights;
    shared_problem.routed_scaling_factor = p.routed_scaling_factor;
    shared_problem.seq_len = p.seq_len;
    shared_problem.stream = p.stream;
    shared_problem.output = p.output;

    kernel6_internal::Gemm2Workspace shared_workspace{};
    shared_workspace.gemm2_output = workspace.gemm2_output;
    shared_workspace.output_accum = workspace.output_accum;
    shared_workspace.cutlass_workspace = nullptr;
    shared_workspace.cutlass_workspace_bytes = 0;

    return kernel6_internal::launch_fallback_gemm2_combine(
        shared_problem, shared_workspace, total_tokens);
}

cudaError_t launch_tiled_backend(const Kernel4Problem& p,
                                 const Kernel4Workspace& workspace,
                                 int total_tokens,
                                 bool gemm1_only) {
    int local_host_offsets[NUM_LOCAL_EXPERTS + 1];
    const int* host_offsets = p.host_expert_token_offsets;
    if (!host_offsets) {
        CUDA_CHECK(cudaMemcpy(
            local_host_offsets,
            p.expert_token_offsets,
            sizeof(local_host_offsets),
            cudaMemcpyDeviceToHost));
        host_offsets = local_host_offsets;
    }

    dim3 block_gemm1(BN);
    dim3 block_micro(K4_MICRO_BN);
    dim3 block_small(K4_SMALL_BN);
    for (int expert = 0; expert < NUM_LOCAL_EXPERTS; ++expert) {
        int begin = host_offsets[expert];
        int end = host_offsets[expert + 1];
        int token_count = end - begin;
        if (token_count <= 0) {
            continue;
        }

        // hidden_states stays in original sequence order; token_indices_e
        // already remaps dispatched rows back to original tokens.
        const fp8_e4m3* act_e = p.hidden_states;
        const int* token_indices_e = p.token_indices + begin;
        const fp8_e4m3* w_e = p.gemm1_weights +
            (size_t)expert * GEMM1_OUT_SIZE * HIDDEN_SIZE;
        const float* ws_e = p.gemm1_weights_scale +
            (size_t)expert * NUM_GEMM1_OUT_BLOCKS * NUM_HIDDEN_BLOCKS;
        __nv_bfloat16* out_e = workspace.gemm1_output + (size_t)begin * INTERMEDIATE_SIZE;

        if (token_count <= 4) {
            dim3 grid_micro((token_count + K4_MICRO_BT - 1) / K4_MICRO_BT, NUM_INTER_BLOCKS);
            fp8_gemm1_swiglu_micro_kernel<<<grid_micro, block_micro, 0, p.stream>>>(
                act_e,
                p.hidden_states_scale,
                token_indices_e,
                token_count,
                w_e,
                ws_e,
                out_e,
                p.seq_len
            );
        } else if (token_count <= 16) {
            dim3 grid_small((token_count + K4_SMALL_BT - 1) / K4_SMALL_BT,
                            INTERMEDIATE_SIZE / K4_SMALL_BN);
            fp8_gemm1_swiglu_small_kernel<<<grid_small, block_small, 0, p.stream>>>(
                act_e,
                p.hidden_states_scale,
                token_indices_e,
                token_count,
                w_e,
                ws_e,
                out_e,
                p.seq_len
            );
        } else {
            dim3 grid_gemm1((token_count + BT - 1) / BT, NUM_INTER_BLOCKS);
            fp8_gemm1_swiglu_kernel<<<grid_gemm1, block_gemm1, 0, p.stream>>>(
                act_e,
                p.hidden_states_scale,
                token_indices_e,
                token_count,
                w_e,
                ws_e,
                out_e,
                p.seq_len
            );
        }
        CUDA_CHECK(cudaGetLastError());
    }

    if (gemm1_only) {
        return cudaSuccess;
    }

    CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        kernel6_internal::output_accum_bytes(p.seq_len),
        p.stream));

    kernel6_internal::Gemm2Problem shared_problem{};
    shared_problem.hidden_states = workspace.gemm1_output;
    shared_problem.gemm2_weights = p.gemm2_weights;
    shared_problem.gemm2_weights_scale = p.gemm2_weights_scale;
    shared_problem.expert_token_offsets = p.expert_token_offsets;
    shared_problem.token_indices = p.token_indices;
    shared_problem.local_expert_ids = p.local_expert_ids;
    shared_problem.token_expert_weights = p.token_expert_weights;
    shared_problem.routed_scaling_factor = p.routed_scaling_factor;
    shared_problem.seq_len = p.seq_len;
    shared_problem.stream = p.stream;
    shared_problem.output = p.output;

    kernel6_internal::Gemm2Workspace shared_workspace{};
    shared_workspace.gemm2_output = workspace.gemm2_output;
    shared_workspace.output_accum = workspace.output_accum;
    shared_workspace.cutlass_workspace = nullptr;
    shared_workspace.cutlass_workspace_bytes = 0;

    return kernel6_internal::launch_fallback_gemm2_combine(
        shared_problem, shared_workspace, total_tokens);
}

}  // namespace kernel4_internal
