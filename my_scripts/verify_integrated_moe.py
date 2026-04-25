import argparse
import hashlib
import os
import sys

import torch
from torch.utils.cpp_extension import load


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
GPT5_DIR = os.path.join(PROJECT_ROOT, "gpt5")
LIB_PATH = os.path.join(SCRIPT_DIR, "librouter_ffi.so")
FFI_SOURCE_PATH = os.path.join(PROJECT_ROOT, "solution", "cuda", "moe_ffi.cu")

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)


H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
DEFAULT_SEQ_LENGTHS = [
    7,
    1,
    32,
    80,
    901,
    16,
    15,
    14,
    14107,11948,
    62,59,58,57,56,55,54,53,52,
]
_DISPATCH_ROUTER_FUNC = None
_DISPATCH_SCAN_FUNC = None


def ensure_tvm_ffi_loaded() -> None:
    import tvm_ffi

    if os.path.exists(LIB_PATH):
        tvm_ffi.load_module(LIB_PATH)
        ensure_required_ffi_funcs(tvm_ffi)
        return

    alt_path = os.path.abspath("librouter_ffi.so")
    if os.path.exists(alt_path):
        tvm_ffi.load_module(alt_path)
        ensure_required_ffi_funcs(tvm_ffi)
        return

    raise FileNotFoundError(
        f"Could not find librouter_ffi.so in {SCRIPT_DIR} or the current directory. "
        "Build it first with ./my_scripts/build.sh."
    )


def ensure_required_ffi_funcs(tvm_ffi_module) -> None:
    required = [
        "router_ffi",
        "scan_ffi",
        "reindex_ffi",
        "gemm1_swiglu_ffi",
        "kernel6_ffi",
        "integrated_moe_ffi",
    ]
    missing = [
        name
        for name in required
        if tvm_ffi_module.get_global_func(name, allow_missing=True) is None
    ]
    if missing:
        rebuild_hint = (
            f"Missing TVM FFI functions: {', '.join(missing)}. "
            f"The loaded library at {LIB_PATH} looks stale. "
            "Please rebuild it with `./my_scripts/build.sh` and rerun the verifier."
        )
        if os.path.exists(LIB_PATH) and os.path.exists(FFI_SOURCE_PATH):
            lib_mtime = os.path.getmtime(LIB_PATH)
            src_mtime = os.path.getmtime(FFI_SOURCE_PATH)
            if lib_mtime < src_mtime:
                rebuild_hint += " The source file is newer than the shared library."
        raise RuntimeError(rebuild_hint)


def load_gold_kernel():
    main_cpp = os.path.join(GPT5_DIR, "main.cpp")
    kernel_cu = os.path.join(GPT5_DIR, "kernel.cu")
    kernel_h = os.path.join(GPT5_DIR, "kernel.h")
    for path in (main_cpp, kernel_cu, kernel_h):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing gold kernel source: {path}")

    hasher = hashlib.sha1()
    with open(main_cpp, "rb") as f:
        hasher.update(f.read())
    with open(kernel_cu, "rb") as f:
        hasher.update(f.read())
    hasher.update(torch.__version__.encode("utf-8"))
    hasher.update((torch.version.cuda or "cpu").encode("utf-8"))
    key = hasher.hexdigest()[:10]

    module_name = f"gpt5_moe_gold_{key}"
    return load(
        name=module_name,
        sources=[main_cpp, kernel_cu],
        extra_include_paths=[GPT5_DIR],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def make_generator(seed: int, seq_len: int, salt: int, device: str) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + seq_len * 1009 + salt)
    return gen


def randn_to_float8_chunked(
    shape: tuple[int, ...],
    *,
    device: str,
    generator: torch.Generator,
    scale: float,
    chunk_dim: int = 0,
    chunk_size: int = 1,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    out = torch.empty(shape, device=device, dtype=torch.float8_e4m3fn)
    for start in range(0, shape[chunk_dim], chunk_size):
        stop = min(start + chunk_size, shape[chunk_dim])
        chunk_shape = list(shape)
        chunk_shape[chunk_dim] = stop - start
        chunk = torch.randn(
            tuple(chunk_shape),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        if scale != 1.0:
            chunk.mul_(scale)
        out[start:stop].copy_(chunk.to(torch.float8_e4m3fn))
        del chunk
    return out


def make_static_inputs(device: str, seed: int) -> dict[str, torch.Tensor]:
    gemm1_weights = randn_to_float8_chunked(
        (E_LOCAL, 2 * I, H),
        device=device,
        generator=make_generator(seed, 0, 11, device),
        scale=0.05,
    )
    gemm1_weights_scale = torch.rand(
        E_LOCAL,
        (2 * I) // 128,
        H // 128,
        device=device,
        dtype=torch.float32,
        generator=make_generator(seed, 0, 13, device),
    )
    gemm2_weights = randn_to_float8_chunked(
        (E_LOCAL, H, I),
        device=device,
        generator=make_generator(seed, 0, 17, device),
        scale=0.05,
    )
    gemm2_weights_scale = torch.rand(
        E_LOCAL,
        H // 128,
        I // 128,
        device=device,
        dtype=torch.float32,
        generator=make_generator(seed, 0, 19, device),
    )
    routing_bias = (
        torch.randn(
            E_GLOBAL,
            device=device,
            dtype=torch.float32,
            generator=make_generator(seed, 0, 23, device),
        )
        * 0.1
    ).to(torch.bfloat16)

    return {
        "routing_bias": routing_bias,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
    }


def make_dynamic_inputs(seq_len: int, device: str, seed: int) -> dict[str, torch.Tensor]:
    routing_logits = (
        torch.randn(
            seq_len,
            E_GLOBAL,
            device=device,
            dtype=torch.float32,
            generator=make_generator(seed, seq_len, 29, device),
        )
        * 0.1
    )
    hidden_states = randn_to_float8_chunked(
        (seq_len, H),
        device=device,
        generator=make_generator(seed, seq_len, 31, device),
        scale=0.1,
        chunk_size=1024,
    )
    hidden_states_scale = torch.rand(
        H // 128,
        seq_len,
        device=device,
        dtype=torch.float32,
        generator=make_generator(seed, seq_len, 37, device),
    )
    return {
        "routing_logits": routing_logits,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
    }


def parse_seq_lengths(args: argparse.Namespace) -> list[int]:
    if args.step <= 0:
        raise ValueError("--step must be positive.")
    if args.seq_lengths:
        seq_lengths = args.seq_lengths
    elif args.use_range:
        seq_lengths = list(range(args.start, args.end + 1, args.step))
    else:
        seq_lengths = list(DEFAULT_SEQ_LENGTHS)
    if not seq_lengths:
        raise ValueError("No sequence lengths selected.")
    for seq_len in seq_lengths:
        if seq_len < 1 or seq_len > 32000:
            raise ValueError(f"seq_len={seq_len} is outside the supported range [1, 32000].")
    return seq_lengths


def summarize_diff(gold: torch.Tensor, test: torch.Tensor) -> tuple[float, float, float]:
    gold_f32 = gold.to(device="cpu", dtype=torch.float32, copy=False)
    test_f32 = test.to(device="cpu", dtype=torch.float32, copy=False)
    diff = gold_f32 - test_f32
    mse = torch.mean(diff * diff).item()
    max_abs = torch.max(torch.abs(diff)).item()
    denom = torch.maximum(torch.abs(gold_f32), torch.full_like(gold_f32, 1e-6))
    max_rel = torch.max(torch.abs(diff) / denom).item()
    return mse, max_abs, max_rel


def copy_output_to_pinned_cpu(out: torch.Tensor) -> torch.Tensor:
    out_cpu = torch.empty(out.shape, device="cpu", dtype=torch.float32, pin_memory=True)
    out_cpu.copy_(out, non_blocking=True)
    torch.cuda.current_stream(out.device).synchronize()
    return out_cpu


def get_dispatch_ffi_funcs():
    global _DISPATCH_ROUTER_FUNC, _DISPATCH_SCAN_FUNC
    import tvm_ffi

    if _DISPATCH_ROUTER_FUNC is None:
        _DISPATCH_ROUTER_FUNC = tvm_ffi.get_global_func("router_ffi")
    if _DISPATCH_SCAN_FUNC is None:
        _DISPATCH_SCAN_FUNC = tvm_ffi.get_global_func("scan_ffi")
    return tvm_ffi, _DISPATCH_ROUTER_FUNC, _DISPATCH_SCAN_FUNC


def cutlass_enabled_for_current_device() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 6) and os.path.exists("/workspace/cutlass/include/cutlass/cutlass.h")


@torch.no_grad()
def analyze_dispatch(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    seq_len: int,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> dict[str, int | str]:
    tvm_ffi, router_func, scan_func = get_dispatch_ffi_funcs()
    device = routing_logits.device

    expert_token_counts = torch.zeros(E_GLOBAL, device=device, dtype=torch.int32)
    token_expert_indices = torch.empty(seq_len, 8, device=device, dtype=torch.int32)
    token_expert_weights = torch.empty(seq_len, 8, device=device, dtype=torch.float32)
    token_expert_slots = torch.empty(seq_len, 8, device=device, dtype=torch.int32)

    router_func(
        tvm_ffi.from_dlpack(routing_logits),
        tvm_ffi.from_dlpack(routing_bias),
        tvm_ffi.from_dlpack(expert_token_counts),
        tvm_ffi.from_dlpack(token_expert_indices),
        tvm_ffi.from_dlpack(token_expert_weights),
        tvm_ffi.from_dlpack(token_expert_slots),
        seq_len,
        local_expert_offset,
        routed_scaling_factor,
    )

    expert_token_offsets = torch.zeros(E_LOCAL + 1, device=device, dtype=torch.int32)
    local_counts = expert_token_counts[local_expert_offset : local_expert_offset + E_LOCAL]
    scan_func(
        tvm_ffi.from_dlpack(local_counts),
        tvm_ffi.from_dlpack(expert_token_offsets),
        E_LOCAL + 1,
    )
    torch.cuda.synchronize()

    total_tok = int(expert_token_offsets[-1].item())
    active_experts = int(torch.count_nonzero(local_counts).item())
    cutlass_enabled = cutlass_enabled_for_current_device()

    # Mirror choose_kernel4_backend_policy() in solution/cuda/moe_ffi.cu.
    # The integrated path always passes has_local_expert_ids=true.
    if 1 <= total_tok <= 4:
        k4_backend = "cutlass" if cutlass_enabled else "tiled"
    elif total_tok <= 4:
        k4_backend = "tiled"
    elif cutlass_enabled and 18 <= total_tok <= 256:
        k4_backend = "cutlass"
    elif total_tok <= 256 or seq_len <= 32:
        k4_backend = "fallback"
    elif cutlass_enabled and (total_tok >= 320 or seq_len >= 384):
        k4_backend = "cutlass"
    elif total_tok <= 4096 or seq_len <= 512:
        k4_backend = "tiled"
    else:
        k4_backend = "cutlass" if cutlass_enabled else "tiled"

    if total_tok <= 256 or seq_len <= 64:
        k6_backend = "fallback"
    elif cutlass_enabled and (total_tok >= 768 or seq_len >= 1024):
        k6_backend = "cutlass"
    else:
        k6_backend = "cutlass" if cutlass_enabled else "fallback"

    del expert_token_counts, token_expert_indices, token_expert_weights, token_expert_slots
    del expert_token_offsets, local_counts
    return {
        "total_tok": total_tok,
        "active_experts": active_experts,
        "k4_backend": k4_backend,
        "k6_backend": k6_backend,
    }


def measure_cuda_ms(fn, *args) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


@torch.no_grad()
def verify_one(
    gold_kernel,
    integrated_moe_fn,
    seq_len: int,
    static_inputs: dict[str, torch.Tensor],
    device: str,
    seed: int,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> tuple[bool, dict[str, float]]:
    dynamic_inputs = make_dynamic_inputs(seq_len, device=device, seed=seed)
    common_args = (
        dynamic_inputs["routing_logits"],
        static_inputs["routing_bias"],
        dynamic_inputs["hidden_states"],
        dynamic_inputs["hidden_states_scale"],
        static_inputs["gemm1_weights"],
        static_inputs["gemm1_weights_scale"],
        static_inputs["gemm2_weights"],
        static_inputs["gemm2_weights_scale"],
        local_expert_offset,
        routed_scaling_factor,
    )
    dispatch_stats = analyze_dispatch(
        routing_logits=dynamic_inputs["routing_logits"],
        routing_bias=static_inputs["routing_bias"],
        seq_len=seq_len,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
    )

    gold_out, gold_ms = measure_cuda_ms(gold_kernel.run, *common_args)
    
    # DEBUG: Check if gold even has tokens
    def get_gold_tok_sim(logits, bias, local_offset):
        scores = torch.sigmoid(logits.to(torch.float32) + bias.to(torch.float32))
        T = logits.shape[0]
        count = 0
        for t in range(T):
            group_scores = []
            for g in range(8):
                g_scores = scores[t, g*32 : (g+1)*32]
                top2_sum = torch.topk(g_scores, 2).values.sum()
                group_scores.append(top2_sum.item())
            group_scores = torch.tensor(group_scores)
            top4_groups = torch.topk(group_scores, 4).indices.tolist()
            mask = torch.zeros(256, dtype=torch.bool)
            for g in top4_groups: mask[g*32 : (g+1)*32] = True
            token_scores = torch.where(mask, scores[t].cpu(), torch.tensor(-1e20))
            top8 = torch.topk(token_scores, 8).indices
            count += ((top8 >= local_offset) & (top8 < local_offset + 32)).sum().item()
        return count

    gold_sim_tok = get_gold_tok_sim(dynamic_inputs["routing_logits"], static_inputs["routing_bias"], local_expert_offset)
    # print(f"DEBUG: seq_len={seq_len} gold_sim_tok={gold_sim_tok}")
    gold_out_cpu = copy_output_to_pinned_cpu(gold_out)
    del gold_out
    torch.cuda.empty_cache()

    test_out, test_ms = measure_cuda_ms(integrated_moe_fn, *common_args)
    test_out_cpu = copy_output_to_pinned_cpu(test_out)
    del test_out
    torch.cuda.empty_cache()

    mse, max_abs, max_rel = summarize_diff(gold_out_cpu, test_out_cpu)
    del gold_out_cpu, test_out_cpu
    torch.cuda.empty_cache()
    passed = mse <= 1e-4 and max_abs <= 5e-2
    speedup = (gold_ms / test_ms) if test_ms > 0 else float("inf")
    stats = {
        "seq_len": float(seq_len),
        "mse": mse,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "gpt5_ms": gold_ms,
        "librouter_ffi_ms": test_ms,
        "speedup_vs_gpt5": speedup,
        "total_tok": float(dispatch_stats["total_tok"]),
        "active_experts": float(dispatch_stats["active_experts"]),
        "k4_backend": dispatch_stats["k4_backend"],
        "k6_backend": dispatch_stats["k6_backend"],
        "gold_sim_tok": float(gold_sim_tok),
    }
    for tensor in dynamic_inputs.values():
        del tensor
    torch.cuda.empty_cache()
    return passed, stats


def format_result(passed: bool, stats: dict[str, float]) -> str:
    status = "PASS" if passed else "FAIL"
    return (
        f"[{status}] seq_len={int(stats['seq_len']):5d} "
        f"total_tok={int(stats['total_tok']):6d} "
        f"gold_sim={int(stats['gold_sim_tok']):6d} "
        f"active_experts={int(stats['active_experts']):2d} "
        f"k4={stats['k4_backend']} "
        f"k6={stats['k6_backend']} "
        f"gpt5={stats['gpt5_ms']:.3f}ms "
        f"librouter_ffi={stats['librouter_ffi_ms']:.3f}ms "
        f"speedup={stats['speedup_vs_gpt5']:.3f}x "
        f"mse={stats['mse']:.3e} "
        f"max_abs={stats['max_abs']:.3e} "
        f"max_rel={stats['max_rel']:.3e}"
    )


def format_oom_skip(seq_len: int, err: torch.OutOfMemoryError) -> str:
    first_line = str(err).splitlines()[0] if str(err) else "CUDA out of memory."
    return f"[SKIP] seq_len={seq_len:5d} reason={first_line}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify my_scripts/integrated_moe_kernel.py against the gold kernel in gpt5."
    )
    parser.add_argument("--seq-lengths", type=int, nargs="*", help="Explicit sequence lengths to run.")
    parser.add_argument(
        "--use-range",
        action="store_true",
        help="Use the --start/--end/--step range instead of the default 30 targeted testcases.",
    )
    parser.add_argument("--start", type=int, default=1, help="Range start when --use-range is enabled.")
    parser.add_argument("--end", type=int, default=32000, help="Range end when --use-range is enabled.")
    parser.add_argument("--step", type=int, default=1, help="Range step when --use-range is enabled.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--device", default="cuda", help="Torch device to use. CUDA is required.")
    parser.add_argument("--local-expert-offset", type=int, default=96, help="Local expert offset for both kernels.")
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0, help="Routing scaling factor.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failing sequence length.")
    parser.add_argument(
        "--fail-on-oom",
        action="store_true",
        help="Abort immediately if any testcase runs out of CUDA memory instead of skipping it.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device != "cuda":
        raise ValueError("This verifier currently expects --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    ensure_tvm_ffi_loaded()
    integrated_moe_fn = __import__("integrated_moe_kernel").integrated_moe
    gold_kernel = load_gold_kernel()
    seq_lengths = parse_seq_lengths(args)
    print(f"Running {len(seq_lengths)} testcase(s): {seq_lengths}", flush=True)
    static_inputs = make_static_inputs(device=args.device, seed=args.seed)

    all_stats: list[dict[str, float]] = []
    failures: list[dict[str, float]] = []
    skipped_oom: list[int] = []
    for seq_len in seq_lengths:
        try:
            passed, stats = verify_one(
                gold_kernel=gold_kernel,
                integrated_moe_fn=integrated_moe_fn,
                seq_len=seq_len,
                static_inputs=static_inputs,
                device=args.device,
                seed=args.seed,
                local_expert_offset=args.local_expert_offset,
                routed_scaling_factor=args.routed_scaling_factor,
            )
        except torch.OutOfMemoryError as err:
            torch.cuda.empty_cache()
            if args.fail_on_oom:
                raise
            skipped_oom.append(seq_len)
            print(format_oom_skip(seq_len, err), flush=True)
            continue
        all_stats.append(stats)
        print(format_result(passed, stats), flush=True)
        if not passed:
            failures.append(stats)
            if args.stop_on_fail:
                break

    total_gpt5_ms = sum(item["gpt5_ms"] for item in all_stats)
    total_librouter_ffi_ms = sum(item["librouter_ffi_ms"] for item in all_stats)
    overall_speedup = (total_gpt5_ms / total_librouter_ffi_ms) if total_librouter_ffi_ms > 0 else float("inf")
    skipped_summary = f" skipped_oom={skipped_oom}" if skipped_oom else ""

    if failures:
        worst = max(failures, key=lambda item: (item["mse"], item["max_abs"], item["max_rel"]))
        print(
            "\nVerification finished with failures. "
            f"worst_seq_len={int(worst['seq_len'])} "
            f"worst_mse={worst['mse']:.3e} "
            f"worst_max_abs={worst['max_abs']:.3e} "
            f"worst_max_rel={worst['max_rel']:.3e} "
            f"total_gpt5={total_gpt5_ms:.3f}ms "
            f"total_librouter_ffi={total_librouter_ffi_ms:.3f}ms "
            f"overall_speedup={overall_speedup:.3f}x"
            f"{skipped_summary}"
        )
        return 1

    print(
        f"\nVerification succeeded for {len(all_stats)} sequence length(s). "
        f"total_gpt5={total_gpt5_ms:.3f}ms "
        f"total_librouter_ffi={total_librouter_ffi_ms:.3f}ms "
        f"overall_speedup={overall_speedup:.3f}x"
        f"{skipped_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
