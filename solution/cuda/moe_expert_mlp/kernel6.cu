#include "kernel6_internal.cuh"

#include <algorithm>

using namespace moe_spec;

size_t k6_query_workspace(int seq_len,
                          int total_dispatched_tokens,
                          size_t cutlass_workspace_bytes) {
    size_t total = 0;
    total += kernel6_internal::align_up(kernel6_internal::gemm2_output_bytes(total_dispatched_tokens));
    total += kernel6_internal::align_up(kernel6_internal::output_accum_bytes(seq_len));
#if defined(K4_ENABLE_CUTLASS)
    cutlass_workspace_bytes = std::max(
        cutlass_workspace_bytes,
        kernel6_internal::cutlass_aux_bytes(total_dispatched_tokens));
#endif
    total += kernel6_internal::align_up(cutlass_workspace_bytes);
    return total;
}

Kernel6Workspace k6_bind_workspace(void* storage,
                                   size_t storage_bytes,
                                   int seq_len,
                                   int total_dispatched_tokens,
                                   size_t cutlass_workspace_bytes) {
    Kernel6Workspace workspace{};
    workspace.storage = storage;
    workspace.storage_bytes = storage_bytes;
#if defined(K4_ENABLE_CUTLASS)
    cutlass_workspace_bytes = std::max(
        cutlass_workspace_bytes,
        kernel6_internal::cutlass_aux_bytes(total_dispatched_tokens));
#endif
    workspace.cutlass_workspace_bytes = cutlass_workspace_bytes;
    if (!storage) {
        return workspace;
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(storage);
    uintptr_t cursor = kernel6_internal::align_up(base);

    if (total_dispatched_tokens > 0) {
        workspace.gemm2_output = reinterpret_cast<float*>(cursor);
        cursor += kernel6_internal::align_up(
            kernel6_internal::gemm2_output_bytes(total_dispatched_tokens));
    }

    workspace.output_accum = reinterpret_cast<float*>(cursor);
    cursor += kernel6_internal::align_up(kernel6_internal::output_accum_bytes(seq_len));

    if (cutlass_workspace_bytes > 0) {
        workspace.cutlass_workspace = reinterpret_cast<void*>(cursor);
        cursor += kernel6_internal::align_up(cutlass_workspace_bytes);
    }

    if (cursor - base > storage_bytes) {
        Kernel6Workspace invalid{};
        invalid.storage = storage;
        invalid.storage_bytes = storage_bytes;
        return invalid;
    }

    return workspace;
}

bool k6_cutlass_available() {
#if defined(K4_ENABLE_CUTLASS)
    return kernel6_internal::current_device_is_sm86_or_better();
#else
    return false;
#endif
}

cudaError_t k6_launch(const Kernel6Problem& p,
                      const Kernel6Workspace& workspace) {
    if (!p.expert_token_offsets || !p.output) {
        return cudaErrorInvalidValue;
    }

    int total_tokens = p.total_dispatched_tokens;
    if (total_tokens < 0) {
        int last = 0;
        K6_CUDA_CHECK(cudaMemcpy(&last,
            p.expert_token_offsets + NUM_LOCAL_EXPERTS,
            sizeof(int), cudaMemcpyDeviceToHost));
        total_tokens = last;
    }

    if (total_tokens == 0) {
        K6_CUDA_CHECK(cudaMemsetAsync(
            p.output,
            0,
            (size_t)p.seq_len * HIDDEN_SIZE * sizeof(__nv_bfloat16),
            p.stream));
        return cudaSuccess;
    }

    if (!p.hidden_states || !p.gemm2_weights || !p.gemm2_weights_scale ||
        !p.token_indices || !p.token_expert_weights) {
        return cudaErrorInvalidValue;
    }

    const size_t required_workspace = k6_query_workspace(
        p.seq_len, total_tokens, workspace.cutlass_workspace_bytes);
    const bool needs_gemm2_output =
        (p.backend == Kernel6Backend::Cutlass) ||
        (p.backend == Kernel6Backend::Auto && k6_cutlass_available());
    if (!workspace.storage || workspace.storage_bytes < required_workspace ||
        (needs_gemm2_output && !workspace.gemm2_output) || !workspace.output_accum) {
        return cudaErrorInvalidValue;
    }

    if (p.backend == Kernel6Backend::Cutlass && !k6_cutlass_available()) {
        return cudaErrorNotSupported;
    }

    kernel6_internal::Gemm2Problem shared_problem{};
    shared_problem.hidden_states = p.hidden_states;
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
    shared_workspace.cutlass_workspace = workspace.cutlass_workspace;
    shared_workspace.cutlass_workspace_bytes = workspace.cutlass_workspace_bytes;

#if defined(K4_ENABLE_CUTLASS)
    if ((p.backend == Kernel6Backend::Cutlass || p.backend == Kernel6Backend::Auto) &&
        k6_cutlass_available()) {
        return kernel6_internal::launch_cutlass_gemm2_combine(
            shared_problem, shared_workspace, total_tokens);
    }
#endif

    if (p.backend == Kernel6Backend::Tiled) {
        return kernel6_internal::launch_tiled_gemm2_combine(
            shared_problem, shared_workspace, total_tokens);
    }

    return kernel6_internal::launch_fallback_gemm2_combine(
        shared_problem, shared_workspace, total_tokens);
}
