// Author:  Eddie Tsai
// Description:  Expert MLP (GEMM1 -> SwiGLU -> GEMM2) with FP8 support
#include "kernel4_internal.cuh"

#include <algorithm>

using namespace moe_spec;

size_t k4_query_workspace(int seq_len,
                          int total_dispatched_tokens,
                          size_t cutlass_workspace_bytes) {
    size_t total = 0;
    total += kernel4_internal::align_up(kernel4_internal::gemm1_output_bytes(total_dispatched_tokens));
    total += kernel4_internal::align_up(kernel6_internal::gemm2_output_bytes(total_dispatched_tokens));
    total += kernel4_internal::align_up(kernel6_internal::output_accum_bytes(seq_len));
#if defined(K4_ENABLE_CUTLASS)
    cutlass_workspace_bytes = std::max(
        cutlass_workspace_bytes,
        kernel4_internal::cutlass_aux_bytes(total_dispatched_tokens));
#endif
    total += kernel4_internal::align_up(cutlass_workspace_bytes);
    return total;
}

Kernel4Workspace k4_bind_workspace(void* storage,
                                   size_t storage_bytes,
                                   int seq_len,
                                   int total_dispatched_tokens,
                                   size_t cutlass_workspace_bytes) {
    Kernel4Workspace workspace{};
    workspace.storage = storage;
    workspace.storage_bytes = storage_bytes;
#if defined(K4_ENABLE_CUTLASS)
    cutlass_workspace_bytes = std::max(
        cutlass_workspace_bytes,
        kernel4_internal::cutlass_aux_bytes(total_dispatched_tokens));
#endif
    workspace.cutlass_workspace_bytes = cutlass_workspace_bytes;
    if (!storage) {
        return workspace;
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(storage);
    uintptr_t cursor = kernel4_internal::align_up(base);

    if (total_dispatched_tokens > 0) {
        workspace.gemm1_output = reinterpret_cast<__nv_bfloat16*>(cursor);
        cursor += kernel4_internal::align_up(
            kernel4_internal::gemm1_output_bytes(total_dispatched_tokens));

        workspace.gemm2_output = reinterpret_cast<float*>(cursor);
        cursor += kernel4_internal::align_up(
            kernel6_internal::gemm2_output_bytes(total_dispatched_tokens));
    }

    workspace.output_accum = reinterpret_cast<float*>(cursor);
    cursor += kernel4_internal::align_up(kernel6_internal::output_accum_bytes(seq_len));

    if (cutlass_workspace_bytes > 0) {
        workspace.cutlass_workspace = reinterpret_cast<void*>(cursor);
        cursor += kernel4_internal::align_up(cutlass_workspace_bytes);
    }

    if (cursor - base > storage_bytes) {
        Kernel4Workspace invalid{};
        invalid.storage = storage;
        invalid.storage_bytes = storage_bytes;
        return invalid;
    }

    return workspace;
}

bool k4_cutlass_available() {
#if defined(K4_ENABLE_CUTLASS)
    return kernel6_internal::current_device_is_sm86_or_better();
#else
    return false;
#endif
}

cudaError_t k4_launch_impl(const Kernel4Problem& p,
                           const Kernel4Workspace& workspace,
                           bool gemm1_only) {
    if (!p.expert_token_offsets || (!gemm1_only && !p.output)) {
        return cudaErrorInvalidValue;
    }

    int total_tokens = p.total_dispatched_tokens;
    if (total_tokens < 0) {
        int last = 0;
        CUDA_CHECK(cudaMemcpy(&last,
            p.expert_token_offsets + NUM_LOCAL_EXPERTS,
            sizeof(int), cudaMemcpyDeviceToHost));
        total_tokens = last;
    }

    if (total_tokens == 0) {
        if (!gemm1_only) {
            CUDA_CHECK(cudaMemsetAsync(
                p.output,
                0,
                (size_t)p.seq_len * HIDDEN_SIZE * sizeof(__nv_bfloat16),
                p.stream));
        }
        return cudaSuccess;
    }

    if (!p.hidden_states || !p.hidden_states_scale || !p.gemm1_weights ||
        !p.gemm1_weights_scale || !p.token_indices || 
        (!gemm1_only && (!p.gemm2_weights || !p.gemm2_weights_scale || !p.token_expert_weights))) {
        return cudaErrorInvalidValue;
    }

    const size_t required_workspace = k4_query_workspace(
        p.seq_len, total_tokens, workspace.cutlass_workspace_bytes);
    if (!workspace.storage || workspace.storage_bytes < required_workspace ||
        !workspace.gemm1_output || (!gemm1_only && (!workspace.gemm2_output || !workspace.output_accum))) {
        return cudaErrorInvalidValue;
    }

    if (p.backend == Kernel4Backend::Cutlass && !k4_cutlass_available()) {
        return cudaErrorNotSupported;
    }

#if defined(K4_ENABLE_CUTLASS)
    if ((p.backend == Kernel4Backend::Cutlass || p.backend == Kernel4Backend::Auto) &&
        k4_cutlass_available()) {
        return kernel4_internal::launch_cutlass_backend(p, workspace, total_tokens, gemm1_only);
    }
#endif

    if (p.backend == Kernel4Backend::Tiled) {
        return kernel4_internal::launch_tiled_backend(p, workspace, total_tokens, gemm1_only);
    }

    return kernel4_internal::launch_fallback_backend(p, workspace, total_tokens, gemm1_only);
}

cudaError_t k4_launch(const Kernel4Problem& p,
                      const Kernel4Workspace& workspace) {
    return k4_launch_impl(p, workspace, false);
}

cudaError_t k4_launch_gemm1(const Kernel4Problem& p,
                            const Kernel4Workspace& workspace) {
    return k4_launch_impl(p, workspace, true);
}
