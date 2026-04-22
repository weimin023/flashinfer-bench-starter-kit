import torch
import tvm_ffi

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
    if total_assigned == 0:
        return torch.empty(0, dtype=torch.int32, device=device)

    local_mask = (
        (token_expert_indices >= local_expert_offset) &
        (token_expert_indices < local_expert_offset + E_LOCAL)
    )
    local_experts = (token_expert_indices[local_mask] - local_expert_offset).to(torch.int64)
    slots = token_expert_slots[local_mask].to(torch.int64)
    dest = expert_token_offsets[local_experts].to(torch.int64) + slots

    token_ids = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, TOP_K)
    token_indices = torch.empty(total_assigned, dtype=torch.int32, device=device)
    token_indices[dest] = token_ids[local_mask]
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
    if total_assigned == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    local_mask = (
        (token_expert_indices >= local_expert_offset) &
        (token_expert_indices < local_expert_offset + E_LOCAL)
    )
    local_experts = (token_expert_indices[local_mask] - local_expert_offset).to(torch.int64)
    slots = token_expert_slots[local_mask].to(torch.int64)
    dest = expert_token_offsets[local_experts].to(torch.int64) + slots

    merged_weights = torch.empty(total_assigned, dtype=torch.float32, device=device)
    merged_weights[dest] = token_expert_weights[local_mask]
    return merged_weights

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
    
    # --- Step 1: ROUTING ---
    torch.cuda.nvtx.range_push("moe_routing")
    expert_token_counts = torch.zeros(E_GLOBAL, device=device, dtype=torch.int32)
    token_expert_indices = torch.zeros(seq_len, TOP_K, device=device, dtype=torch.int32)
    token_expert_weights = torch.zeros(seq_len, TOP_K, device=device, dtype=torch.float32)
    token_expert_slots = torch.zeros(seq_len, TOP_K, device=device, dtype=torch.int32)
    
    router_func = tvm_ffi.get_global_func("router_ffi")
    router_func(
        tvm_ffi.from_dlpack(routing_logits.to(torch.float32)),
        tvm_ffi.from_dlpack(routing_bias.to(torch.bfloat16)),
        tvm_ffi.from_dlpack(expert_token_counts),
        tvm_ffi.from_dlpack(token_expert_indices),
        tvm_ffi.from_dlpack(token_expert_weights),
        tvm_ffi.from_dlpack(token_expert_slots),
        seq_len, local_expert_offset, routed_scaling_factor
    )
    torch.cuda.nvtx.range_pop()
    
    # --- Step 2: SCAN ---
    torch.cuda.nvtx.range_push("moe_scan")
    expert_token_offsets = torch.zeros(E_LOCAL + 1, device=device, dtype=torch.int32)
    local_counts = expert_token_counts[local_expert_offset : local_expert_offset + E_LOCAL]
    
    scan_func = tvm_ffi.get_global_func("scan_ffi")
    scan_func(
        tvm_ffi.from_dlpack(local_counts),
        tvm_ffi.from_dlpack(expert_token_offsets),
        E_LOCAL + 1
    )
    torch.cuda.nvtx.range_pop()
    
    total_assigned = expert_token_offsets[E_LOCAL].item()
    if total_assigned == 0:
        return torch.zeros(seq_len, H, device=device, dtype=torch.bfloat16)

    # --- Step 3: PREPARE INDICES ---
    torch.cuda.nvtx.range_push("moe_reindex")
    token_indices = get_token_indices(token_expert_indices, token_expert_slots, expert_token_offsets, seq_len, TOP_K, E_LOCAL, local_expert_offset)
    merged_token_weights = gather_merged_weights(token_expert_indices, token_expert_weights, token_expert_slots, expert_token_offsets, seq_len, TOP_K, E_LOCAL, local_expert_offset)
    torch.cuda.nvtx.range_pop()

    # --- Step 4: KERNEL 4 (GEMM1 + SwiGLU) ---
    # I = intermediate_size. gemm2_weights shape is [E_LOCAL, H, I]
    I = gemm2_weights.shape[2] 
    inter_tokens = torch.zeros(total_assigned, I, device=device, dtype=torch.bfloat16)
    torch.cuda.nvtx.range_push("moe_gemm1_swiglu")
    gemm1_func = tvm_ffi.get_global_func("gemm1_swiglu_ffi")
    gemm1_func(
        tvm_ffi.from_dlpack(hidden_states),
        tvm_ffi.from_dlpack(hidden_states_scale.to(torch.float32)),
        tvm_ffi.from_dlpack(gemm1_weights),
        tvm_ffi.from_dlpack(gemm1_weights_scale.to(torch.float32)),
        tvm_ffi.from_dlpack(expert_token_offsets),
        tvm_ffi.from_dlpack(token_indices),
        tvm_ffi.from_dlpack(inter_tokens),
        seq_len, local_expert_offset
    )
    torch.cuda.nvtx.range_pop()
    
    # --- Step 5: KERNEL 6 (GEMM2) ---
    output = torch.zeros(seq_len, H, device=device, dtype=torch.bfloat16)
    torch.cuda.nvtx.range_push("moe_gemm2_acc")
    gemm2_func = tvm_ffi.get_global_func("kernel6_ffi")
    gemm2_func(
        tvm_ffi.from_dlpack(inter_tokens),
        tvm_ffi.from_dlpack(gemm2_weights),
        tvm_ffi.from_dlpack(gemm2_weights_scale.to(torch.float32)),
        tvm_ffi.from_dlpack(expert_token_offsets),
        tvm_ffi.from_dlpack(token_indices),
        tvm_ffi.from_dlpack(merged_token_weights),
        tvm_ffi.from_dlpack(output),
        seq_len, local_expert_offset, routed_scaling_factor
    )
    torch.cuda.nvtx.range_pop()
    
    return output
