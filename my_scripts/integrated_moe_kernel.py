import torch
import tvm_ffi


_ROUTER_FUNC = None
_SCAN_FUNC = None
_REINDEX_FUNC = None
_GEMM1_FUNC = None
_GEMM2_FUNC = None
_INTEGRATED_MOE_FUNC = None


def _get_global_func_cached(name: str):
    global _ROUTER_FUNC, _SCAN_FUNC, _REINDEX_FUNC, _GEMM1_FUNC, _GEMM2_FUNC, _INTEGRATED_MOE_FUNC
    cache = {
        "router_ffi": "_ROUTER_FUNC",
        "scan_ffi": "_SCAN_FUNC",
        "reindex_ffi": "_REINDEX_FUNC",
        "gemm1_swiglu_ffi": "_GEMM1_FUNC",
        "kernel6_ffi": "_GEMM2_FUNC",
        "integrated_moe_ffi": "_INTEGRATED_MOE_FUNC",
    }
    cache_name = cache[name]
    func = globals()[cache_name]
    if func is None:
        func = tvm_ffi.get_global_func(name)
        globals()[cache_name] = func
    return func

def get_token_indices(
    token_expert_indices: torch.Tensor,
    token_expert_slots: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    seq_len: int,
    TOP_K: int,
    E_LOCAL: int,
    local_expert_offset: int
) -> torch.Tensor:
    """
    Computes the reverse mapping: dispatched_slot -> original_token_idx.
    
    Args:
        token_expert_indices: [seq_len, TOP_K]
        token_expert_slots: [seq_len, TOP_K]
        expert_token_offsets: [E_LOCAL + 1]
        seq_len: Total tokens in batch
        TOP_K: Experts per token
        E_LOCAL: Experts on this rank
        local_expert_offset: Global offset for local experts
        
    Returns:
        token_indices: [total_assigned] (int32)
    """
    device = token_expert_indices.device
    total_assigned = expert_token_offsets[E_LOCAL].item()
    
    token_indices = torch.zeros(total_assigned, dtype=torch.int32, device=device)
    
    indices_cpu = token_expert_indices.cpu()
    slots_cpu = token_expert_slots.cpu()
    offsets_cpu = expert_token_offsets.cpu()
    
    for t in range(seq_len):
        for k in range(TOP_K):
            ge = indices_cpu[t, k].item()
            if local_expert_offset <= ge < local_expert_offset + E_LOCAL:
                le = ge - local_expert_offset
                slot = slots_cpu[t, k].item()
                dest = offsets_cpu[le].item() + slot
                token_indices[dest] = t
                
    return token_indices

def gather_merged_weights(
    token_expert_indices: torch.Tensor,
    token_expert_weights: torch.Tensor,
    token_expert_slots: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    seq_len: int,
    TOP_K: int,
    E_LOCAL: int,
    local_expert_offset: int
) -> torch.Tensor:
    """
    Gathers routing weights corresponding to each dispatched slot.
    
    Args:
        token_expert_indices: [seq_len, TOP_K]
        token_expert_weights: [seq_len, TOP_K]
        token_expert_slots: [seq_len, TOP_K]
        expert_token_offsets: [E_LOCAL + 1]
        seq_len: Total tokens in batch
        TOP_K: Experts per token
        E_LOCAL: Experts on this rank
        local_expert_offset: Global offset for local experts
        
    Returns:
        merged_weights: [total_assigned] (float32)
    """
    device = token_expert_indices.device
    total_assigned = expert_token_offsets[E_LOCAL].item()
    
    merged_weights = torch.zeros(total_assigned, dtype=torch.float32, device=device)
    
    indices_cpu = token_expert_indices.cpu()
    weights_cpu = token_expert_weights.cpu()
    slots_cpu = token_expert_slots.cpu()
    offsets_cpu = expert_token_offsets.cpu()
    
    for t in range(seq_len):
        for k in range(TOP_K):
            ge = indices_cpu[t, k].item()
            if local_expert_offset <= ge < local_expert_offset + E_LOCAL:
                le = ge - local_expert_offset
                slot = slots_cpu[t, k].item()
                dest = offsets_cpu[le].item() + slot
                merged_weights[dest] = weights_cpu[t, k].item()
                
    return merged_weights

def reindex_and_gather_gpu(
    token_expert_indices: torch.Tensor,
    token_expert_weights: torch.Tensor,
    token_expert_slots: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    seq_len: int,
    TOP_K: int,
    local_expert_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build expert-grouped token ids and routing weights without host copies."""
    total_assigned = int(expert_token_offsets[-1].item())
    token_indices = torch.empty(total_assigned, dtype=torch.int32, device=token_expert_indices.device)
    local_expert_ids = torch.empty(total_assigned, dtype=torch.int32, device=token_expert_indices.device)
    merged_token_weights = torch.empty(total_assigned, dtype=torch.float32, device=token_expert_indices.device)

    reindex_func = _get_global_func_cached("reindex_ffi")
    reindex_func(
        tvm_ffi.from_dlpack(token_expert_indices),
        tvm_ffi.from_dlpack(token_expert_weights),
        tvm_ffi.from_dlpack(token_expert_slots),
        tvm_ffi.from_dlpack(expert_token_offsets),
        tvm_ffi.from_dlpack(token_indices),
        tvm_ffi.from_dlpack(local_expert_ids),
        tvm_ffi.from_dlpack(merged_token_weights),
        seq_len, local_expert_offset
    )
    return token_indices, local_expert_ids, merged_token_weights

def integrated_moe(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float = 1.0,
    E_GLOBAL: int = 256,
    E_LOCAL: int = 32,
    TOP_K: int = 8
) -> torch.Tensor:
    """
    Integrated DeepSeek-V3 MoE Kernel utilizing TVM FFI.
    Aligned with inout.txt specifications.
    
    Inputs:
        routing_logits: float32 [seq_len, E_GLOBAL]
        routing_bias: bfloat16 [E_GLOBAL]
        hidden_states: float8_e4m3fn [seq_len, H]
        hidden_states_scale: float32 [num_hidden_blocks, seq_len]
        gemm1_weights: float8_e4m3fn [E_LOCAL, 2*I, H]
        gemm1_weights_scale: float32 [E_LOCAL, num_gemm1_out_blocks, num_hidden_blocks]
        gemm2_weights: float8_e4m3fn [E_LOCAL, H, I]
        gemm2_weights_scale: float32 [E_LOCAL, num_hidden_blocks, num_intermediate_blocks]
        local_expert_offset: int32 Scalar
        routed_scaling_factor: float32 Scalar
        
    Constants:
        E_GLOBAL: Total experts (default 256)
        E_LOCAL: Locally computed experts (default 32)
        TOP_K: Active experts per token (default 8)
        
    Returns:
        output: bfloat16 [seq_len, H]
    """
    device = routing_logits.device
    seq_len = routing_logits.shape[0]
    H = hidden_states.shape[1]
    output = torch.zeros(seq_len, H, device=device, dtype=torch.bfloat16)

    routing_logits_f32 = routing_logits if routing_logits.dtype == torch.float32 else routing_logits.to(torch.float32)
    routing_bias_bf16 = routing_bias if routing_bias.dtype == torch.bfloat16 else routing_bias.to(torch.bfloat16)
    hidden_states_scale_f32 = (
        hidden_states_scale if hidden_states_scale.dtype == torch.float32 else hidden_states_scale.to(torch.float32)
    )
    gemm1_weights_scale_f32 = (
        gemm1_weights_scale if gemm1_weights_scale.dtype == torch.float32 else gemm1_weights_scale.to(torch.float32)
    )
    gemm2_weights_scale_f32 = (
        gemm2_weights_scale if gemm2_weights_scale.dtype == torch.float32 else gemm2_weights_scale.to(torch.float32)
    )

    torch.cuda.nvtx.range_push("integrated_moe_ffi")
    integrated_func = _get_global_func_cached("integrated_moe_ffi")
    integrated_func(
        tvm_ffi.from_dlpack(routing_logits_f32),
        tvm_ffi.from_dlpack(routing_bias_bf16),
        tvm_ffi.from_dlpack(hidden_states),
        tvm_ffi.from_dlpack(hidden_states_scale_f32),
        tvm_ffi.from_dlpack(gemm1_weights),
        tvm_ffi.from_dlpack(gemm1_weights_scale_f32),
        tvm_ffi.from_dlpack(gemm2_weights),
        tvm_ffi.from_dlpack(gemm2_weights_scale_f32),
        tvm_ffi.from_dlpack(output),
        seq_len,
        local_expert_offset,
        routed_scaling_factor,
    )
    torch.cuda.nvtx.range_pop()

    return output
