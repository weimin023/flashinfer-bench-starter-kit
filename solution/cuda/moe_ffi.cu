#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/tensor.h>
#include <algorithm>

// Include the existing CUDA kernel implementations
#include "moe_routing.cu"
#include "moe_scan.cu"
#include "moe_dispatch.cu"
#include "moe_expert_mlp/kernel4_cuda_kernels.cu"
#include "moe_expert_mlp/kernel4_backends.cu"
#include "moe_expert_mlp/kernel4.cu"
#include "moe_expert_mlp/kernel6_cuda_kernels.cu"
#include "moe_expert_mlp/kernel6_backends.cu"
#include "moe_expert_mlp/kernel6.cu"





namespace ffi = tvm::ffi;

namespace {

struct PersistentBuffer {
    void* ptr{nullptr};
    size_t bytes{0};

    ~PersistentBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    void ensure(size_t required_bytes) {
        if (required_bytes == 0 || required_bytes <= bytes) {
            return;
        }
        if (ptr) {
            cudaFree(ptr);
        }
        cudaMalloc(&ptr, required_bytes);
        bytes = required_bytes;
    }
};

PersistentBuffer& scan_workspace() {
    static PersistentBuffer buffer;
    return buffer;
}

PersistentBuffer& k4_workspace_cache() {
    static PersistentBuffer buffer;
    return buffer;
}

PersistentBuffer& k6_workspace_cache() {
    static PersistentBuffer buffer;
    return buffer;
}

__global__ void finalize_scan_offsets_kernel(const int* counts, const int* offsets, int* output, int num_counts) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[num_counts] = offsets[num_counts - 1] + counts[num_counts - 1];
    }
}

}  // namespace

// ─── Router FFI ───────────────────────────────────────────────────────────────

void router_ffi_wrapper(ffi::Tensor routing_logits,         // [T, 256]
                        ffi::Tensor routing_bias,           // [256]
                        ffi::Tensor expert_token_counts,    // [256]
                        ffi::Tensor token_expert_indices,   // [T, 8]
                        ffi::Tensor token_expert_weights,   // [T, 8]
                        ffi::Tensor token_expert_slots,     // [T, 8]
                        int T, int local_expert_offset, float routed_scaling_factor) {
    
    const int E_GLOBAL = 256;
    const int E_LOCAL = 32;
    const int TOP_K = 8;
    
    dim3 threads(256);
    dim3 blocks(T);
    
    router<E_GLOBAL, E_LOCAL, TOP_K><<<blocks, threads>>>(
        static_cast<const float*>(routing_logits.data_ptr()),
        static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
        static_cast<int*>(expert_token_counts.data_ptr()),
        static_cast<int*>(token_expert_indices.data_ptr()),
        static_cast<float*>(token_expert_weights.data_ptr()),
        static_cast<int*>(token_expert_slots.data_ptr()),
        T, local_expert_offset, routed_scaling_factor
    );
}

static auto _router = ffi::reflection::GlobalDef().def("router_ffi", router_ffi_wrapper);

// ─── Scan (Prefix Sum) FFI ───────────────────────────────────────────────────

void scan_ffi_wrapper(ffi::Tensor counts, ffi::Tensor offsets, int num_items) {
    // num_items is E_LOCAL + 1 (e.g., 33)
    // counts has E_LOCAL items (e.g., 32)
    int E = num_items - 1;
    auto& workspace = scan_workspace();
    const size_t temp_storage_bytes = get_scan_temp_storage_bytes(E);
    workspace.ensure(temp_storage_bytes);
    exclusive_scan_cub_persistent(
        workspace.ptr,
        temp_storage_bytes,
        static_cast<int*>(counts.data_ptr()),
        static_cast<int*>(offsets.data_ptr()),
        E);
    finalize_scan_offsets_kernel<<<1, 1>>>(
        static_cast<const int*>(counts.data_ptr()),
        static_cast<const int*>(offsets.data_ptr()),
        static_cast<int*>(offsets.data_ptr()),
        E);
}

static auto _scan = ffi::reflection::GlobalDef().def("scan_ffi", scan_ffi_wrapper);

// ─── Dispatch (Permutation) FFI ──────────────────────────────────────────────

void dispatch_ffi_wrapper(ffi::Tensor hidden_states_fp8,     // [T, H]  FP8 E4M3
                          ffi::Tensor hidden_states_scale,   // [H/128, T]  float32
                          ffi::Tensor token_expert_indices,  // [T, TOP_K]
                          ffi::Tensor token_expert_slots,    // [T, TOP_K]
                          ffi::Tensor expert_offsets,        // [E_GLOBAL + 1]
                          ffi::Tensor permuted_tokens,       // [total, H]
                          int T, int TOP_K, int H) {
    launch_token_dispatch(
        static_cast<const __nv_fp8_storage_t*>(hidden_states_fp8.data_ptr()),
        static_cast<const float*>(hidden_states_scale.data_ptr()),
        static_cast<const int*>(token_expert_indices.data_ptr()),
        static_cast<const int*>(token_expert_slots.data_ptr()),
        static_cast<const int*>(expert_offsets.data_ptr()),
        static_cast<float*>(permuted_tokens.data_ptr()),
        T, TOP_K, H
    );
}

static auto _dispatch = ffi::reflection::GlobalDef().def("dispatch_ffi", dispatch_ffi_wrapper);

// ─── Kernel4 FFI ─────────────────────────────────────────────────────────────

void kernel4_ffi_wrapper(ffi::Tensor hidden_states,
                         ffi::Tensor hidden_states_scale,
                         ffi::Tensor gemm1_weights,
                         ffi::Tensor gemm1_weights_scale,
                         ffi::Tensor gemm2_weights,
                         ffi::Tensor gemm2_weights_scale,
                         ffi::Tensor expert_token_offsets,
                         ffi::Tensor token_indices,
                         ffi::Tensor token_expert_weights,
                         ffi::Tensor output,
                         int seq_len, int local_expert_offset, float routed_scaling_factor) {
                         
    int total_tok = 0;
    cudaMemcpy(&total_tok, static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS, sizeof(int), cudaMemcpyDeviceToHost);

    size_t workspace_bytes = k4_query_workspace(seq_len, total_tok, 0);
    auto& workspace_cache = k4_workspace_cache();
    workspace_cache.ensure(workspace_bytes);

    Kernel4Workspace workspace = k4_bind_workspace(workspace_cache.ptr, workspace_bytes, seq_len, total_tok, 0);

    Kernel4Problem problem{};
    problem.routing_logits = nullptr;
    problem.routing_bias = nullptr;
    problem.seq_len = seq_len;
    problem.hidden_states = static_cast<const fp8_e4m3*>(hidden_states.data_ptr());
    problem.hidden_states_scale = static_cast<const float*>(hidden_states_scale.data_ptr());
    problem.gemm1_weights = static_cast<const fp8_e4m3*>(gemm1_weights.data_ptr());
    problem.gemm1_weights_scale = static_cast<const float*>(gemm1_weights_scale.data_ptr());
    problem.gemm2_weights = static_cast<const fp8_e4m3*>(gemm2_weights.data_ptr());
    problem.gemm2_weights_scale = static_cast<const float*>(gemm2_weights_scale.data_ptr());
    problem.local_expert_offset = local_expert_offset;
    problem.routed_scaling_factor = routed_scaling_factor;
    problem.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    problem.token_indices = static_cast<const int*>(token_indices.data_ptr());
    problem.token_expert_weights = static_cast<const float*>(token_expert_weights.data_ptr());
    problem.output = static_cast<__nv_bfloat16*>(output.data_ptr());
    problem.backend = Kernel4Backend::Auto;
    problem.stream = nullptr;

    k4_launch(problem, workspace);
}

static auto _kernel4 = ffi::reflection::GlobalDef().def("kernel4_ffi", kernel4_ffi_wrapper);

// ─── GEMM1+SwiGLU FFI ───────────────────────────────────────────────────────

void gemm1_swiglu_ffi_wrapper(ffi::Tensor hidden_states,
                              ffi::Tensor hidden_states_scale,
                              ffi::Tensor gemm1_weights,
                              ffi::Tensor gemm1_weights_scale,
                              ffi::Tensor expert_token_offsets,
                              ffi::Tensor token_indices,
                              ffi::Tensor output,
                              int seq_len, int local_expert_offset) {
                         
    int total_tok = 0;
    cudaMemcpy(&total_tok, static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS, sizeof(int), cudaMemcpyDeviceToHost);

    size_t workspace_bytes = k4_query_workspace(seq_len, total_tok, 0);
    auto& workspace_cache = k4_workspace_cache();
    workspace_cache.ensure(workspace_bytes);

    Kernel4Workspace workspace = k4_bind_workspace(workspace_cache.ptr, workspace_bytes, seq_len, total_tok, 0);

    Kernel4Problem problem{};
    problem.seq_len = seq_len;
    problem.hidden_states = static_cast<const fp8_e4m3*>(hidden_states.data_ptr());
    problem.hidden_states_scale = static_cast<const float*>(hidden_states_scale.data_ptr());
    problem.gemm1_weights = static_cast<const fp8_e4m3*>(gemm1_weights.data_ptr());
    problem.gemm1_weights_scale = static_cast<const float*>(gemm1_weights_scale.data_ptr());
    problem.local_expert_offset = local_expert_offset;
    problem.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    problem.token_indices = static_cast<const int*>(token_indices.data_ptr());
    problem.output = nullptr; // Not used for gemm1 call
    problem.backend = Kernel4Backend::Auto;
    problem.stream = nullptr;

    k4_launch_gemm1(problem, workspace);

    // Copy intermediate output from workspace to the provided output tensor
    cudaMemcpy(output.data_ptr(), workspace.gemm1_output, 
               (size_t)total_tok * moe_spec::INTERMEDIATE_SIZE * sizeof(__nv_bfloat16), 
               cudaMemcpyDeviceToDevice);
}

static auto _gemm1_swiglu = ffi::reflection::GlobalDef().def("gemm1_swiglu_ffi", gemm1_swiglu_ffi_wrapper);

// ─── Kernel6 FFI ─────────────────────────────────────────────────────────────

void kernel6_ffi_wrapper(ffi::Tensor hidden_states,
                         ffi::Tensor gemm2_weights,
                         ffi::Tensor gemm2_weights_scale,
                         ffi::Tensor expert_token_offsets,
                         ffi::Tensor token_indices,
                         ffi::Tensor token_expert_weights,
                         ffi::Tensor output,
                         int seq_len, int local_expert_offset, float routed_scaling_factor) {
                         
    int total_tok = 0;
    cudaMemcpy(&total_tok, static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS, sizeof(int), cudaMemcpyDeviceToHost);

    size_t workspace_bytes = k6_query_workspace(seq_len, total_tok, 0);
    auto& workspace_cache = k6_workspace_cache();
    workspace_cache.ensure(workspace_bytes);

    Kernel6Workspace workspace = k6_bind_workspace(workspace_cache.ptr, workspace_bytes, seq_len, total_tok, 0);

    Kernel6Problem problem{};
    problem.hidden_states = static_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
    problem.seq_len = seq_len;
    problem.gemm2_weights = static_cast<const fp8_e4m3*>(gemm2_weights.data_ptr());
    problem.gemm2_weights_scale = static_cast<const float*>(gemm2_weights_scale.data_ptr());
    problem.local_expert_offset = local_expert_offset;
    problem.routed_scaling_factor = routed_scaling_factor;
    problem.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    problem.token_indices = static_cast<const int*>(token_indices.data_ptr());
    problem.token_expert_weights = static_cast<const float*>(token_expert_weights.data_ptr());
    problem.output = static_cast<__nv_bfloat16*>(output.data_ptr());
    problem.backend = Kernel6Backend::Auto;
    problem.stream = nullptr;

    k6_launch(problem, workspace);
}

static auto _kernel6 = ffi::reflection::GlobalDef().def("kernel6_ffi", kernel6_ffi_wrapper);

// ─── MoE Integration Function ──────────────────────────────────────────────────

// Expose a combined wrapper integrating the kernels
void moe_forward_ffi_wrapper(ffi::Tensor routing_logits,
                             ffi::Tensor routing_bias,
                             ffi::Tensor hidden_states,
                             ffi::Tensor hidden_states_scale,
                             ffi::Tensor gemm1_weights,
                             ffi::Tensor gemm1_weights_scale,
                             ffi::Tensor gemm2_weights,
                             ffi::Tensor gemm2_weights_scale,
                             ffi::Tensor expert_token_counts,
                             ffi::Tensor token_expert_indices,
                             ffi::Tensor token_expert_weights,
                             ffi::Tensor token_expert_slots,
                             ffi::Tensor expert_token_offsets,
                             ffi::Tensor token_indices,
                             ffi::Tensor merged_token_weights,
                             ffi::Tensor output,
                             int seq_len, int local_expert_offset, float routed_scaling_factor) {
    // 1. Route
    router_ffi_wrapper(routing_logits, routing_bias, expert_token_counts, 
                       token_expert_indices, token_expert_weights, token_expert_slots, 
                       seq_len, local_expert_offset, routed_scaling_factor);
    cudaDeviceSynchronize();
                       
    // 2. Scan (only for local experts)
    int* local_counts_ptr = static_cast<int*>(expert_token_counts.data_ptr()) + local_expert_offset;
    exclusive_scan_cub(local_counts_ptr, static_cast<int*>(expert_token_offsets.data_ptr()), moe_spec::NUM_LOCAL_EXPERTS);
    cudaDeviceSynchronize();

    // Compute total_tok for the 33rd offset
    int last_c, last_o;
    cudaMemcpy(&last_c, local_counts_ptr + moe_spec::NUM_LOCAL_EXPERTS - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_o, static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int total_tok = last_c + last_o;
    cudaMemcpy(static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS, &total_tok, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    size_t k4_ws_bytes = k4_query_workspace(seq_len, total_tok, 0);

    void* d_k4_ws = nullptr;
    if (k4_ws_bytes > 0) {
        if (cudaMalloc(&d_k4_ws, k4_ws_bytes) != cudaSuccess) {
            printf("CUDA malloc failed for k4_ws\n");
            return;
        }
    }
    
    Kernel4Workspace k4_ws = k4_bind_workspace(d_k4_ws, k4_ws_bytes, seq_len, total_tok, 0);


    
    Kernel4Problem k4_p{};
    k4_p.seq_len = seq_len;
    k4_p.hidden_states = static_cast<const fp8_e4m3*>(hidden_states.data_ptr());
    k4_p.hidden_states_scale = static_cast<const float*>(hidden_states_scale.data_ptr());
    k4_p.gemm1_weights = static_cast<const fp8_e4m3*>(gemm1_weights.data_ptr());
    k4_p.gemm1_weights_scale = static_cast<const float*>(gemm1_weights_scale.data_ptr());
    k4_p.local_expert_offset = local_expert_offset;
    k4_p.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    k4_p.token_indices = static_cast<const int*>(token_indices.data_ptr());
    k4_p.backend = Kernel4Backend::Auto;
    
    k4_launch_gemm1(k4_p, k4_ws);
    cudaDeviceSynchronize();
    
    // 4. Compute Expert MLP Part 2: GEMM2 + Accumulation (Kernel 6)

    size_t k6_ws_bytes = k6_query_workspace(seq_len, total_tok, 0);
    void* d_k6_ws = nullptr;
    if (k6_ws_bytes > 0) {
        if (cudaMalloc(&d_k6_ws, k6_ws_bytes) != cudaSuccess) {
            printf("CUDA malloc failed for k6_ws\n");
            if (d_k4_ws) cudaFree(d_k4_ws);
            return;
        }
    }
    
    Kernel6Workspace k6_ws = k6_bind_workspace(d_k6_ws, k6_ws_bytes, seq_len, total_tok, 0);

    
    Kernel6Problem k6_p{};
    k6_p.hidden_states = k4_ws.gemm1_output;
    k6_p.seq_len = seq_len;
    k6_p.gemm2_weights = static_cast<const fp8_e4m3*>(gemm2_weights.data_ptr());
    k6_p.gemm2_weights_scale = static_cast<const float*>(gemm2_weights_scale.data_ptr());
    k6_p.local_expert_offset = local_expert_offset;
    k6_p.routed_scaling_factor = routed_scaling_factor;
    k6_p.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    k6_p.token_indices = static_cast<const int*>(token_indices.data_ptr());
    k6_p.token_expert_weights = static_cast<const float*>(merged_token_weights.data_ptr());
    k6_p.output = static_cast<__nv_bfloat16*>(output.data_ptr());
    k6_p.backend = Kernel6Backend::Auto;
    
    k6_launch(k6_p, k6_ws);
    cudaDeviceSynchronize();
    
    if (cudaGetLastError() != cudaSuccess) {
        printf("Kernel 6 failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    }


}

static auto _moe_forward = ffi::reflection::GlobalDef().def("moe_forward_ffi", moe_forward_ffi_wrapper);
