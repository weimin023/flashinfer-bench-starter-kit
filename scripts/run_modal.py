"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks on
NVIDIA B200 GPUs via Modal.

This script is configured to match the official MLSys 2026 evaluation software
environment and benchmark settings as closely as Modal allows.

For local development on Modal, resume is disabled by default so code changes
are re-benchmarked even when the dataset volume already contains prior traces.
Pass `resume=True` to restore the official `--resume` behavior.

Other settings:
- Docker image: flashinfer/flashinfer-ci-cu132:20260401-2c675fb
- Isolated runner enabled
- Results saved back to the dataset volume
- Resume is configurable

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution

app = modal.App("flashinfer-bench")

OFFICIAL_DOCKER_IMAGE = "flashinfer/flashinfer-ci-cu132:20260401-2c675fb"
TRACE_VOLUME_NAME = "flashinfer-trace"
TRACE_SET_PATH = "/data"
DEFAULT_TIMEOUT_SECONDS = 300
SAVE_RESULTS = True
DEFAULT_RESUME_RESULTS = False
LOG_LEVEL = "INFO"

COMMON_BENCHMARK_OVERRIDES = {
    "use_isolated_runner": True,
    "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
}

DEFINITION_BENCHMARK_OVERRIDES = {
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": {
        "atol": 1.0,
        "rtol": 0.3,
        "required_matched_ratio": 0.9,
    },
    "gdn_prefill_qk4_v8_d128_k_last": {
        "warmup_runs": 1,
        "iterations": 5,
        "num_trials": 3,
    },
}

trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(OFFICIAL_DOCKER_IMAGE)
    .env({"PYTHONUNBUFFERED": "1"})
    # The published CI image matches the contest software stack, but on Modal the
    # default Python environment may not expose flashinfer_bench as an importable
    # module. Install it explicitly so remote module import succeeds.
    .run_commands("python -m pip install -U flashinfer-bench")
)


def configure_flashinfer_bench_logging(level: str = LOG_LEVEL) -> None:
    """Mirror the CLI's package-level logging setup."""
    logger = logging.getLogger("flashinfer_bench")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False


def build_benchmark_config(solution: Solution):
    """Build benchmark config aligned with the official evaluation profile."""
    from flashinfer_bench.bench import BenchmarkConfig

    overrides = {
        "definitions": [solution.definition],
        "solutions": [solution.name],
        **COMMON_BENCHMARK_OVERRIDES,
    }
    overrides.update(DEFINITION_BENCHMARK_OVERRIDES.get(solution.definition, {}))
    if hasattr(BenchmarkConfig, "default"):
        return BenchmarkConfig.default(**overrides)
    return BenchmarkConfig(**overrides)


def build_equivalent_cli(solution: Solution, resume: bool) -> str:
    """Return the equivalent flashinfer-bench CLI command for the solution."""
    cli = [
        "flashinfer-bench run",
        "--local ./contest-dataset",
        f"--definitions {solution.definition}",
        "--save-results",
        "--use-isolated-runner",
        f"--log-level {LOG_LEVEL}",
        f"--timeout {DEFAULT_TIMEOUT_SECONDS}",
    ]
    if resume:
        cli.append("--resume")
    overrides = DEFINITION_BENCHMARK_OVERRIDES.get(solution.definition, {})
    if "warmup_runs" in overrides:
        cli.append(f"--warmup-runs {overrides['warmup_runs']}")
    if "iterations" in overrides:
        cli.append(f"--iterations {overrides['iterations']}")
    if "num_trials" in overrides:
        cli.append(f"--num-trials {overrides['num_trials']}")
    if "atol" in overrides:
        cli.append(f"--atol {overrides['atol']:g}")
    if "rtol" in overrides:
        cli.append(f"--rtol {overrides['rtol']:g}")
    if "required_matched_ratio" in overrides:
        cli.append(
            "--required-matched-ratio "
            f"{overrides['required_matched_ratio']:g}"
        )
    return " \\\n  ".join(cli)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, resume: bool = DEFAULT_RESUME_RESULTS) -> dict:
    """Run benchmark on Modal B200 and return results."""
    from flashinfer_bench import Benchmark, TraceSet

    configure_flashinfer_bench_logging()
    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    existing_traces = trace_set.traces.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: existing_traces},
    )

    benchmark = Benchmark(bench_trace_set, build_benchmark_config(solution))
    try:
        result_trace_set = benchmark.run_all(SAVE_RESULTS, resume=resume)
    finally:
        benchmark.close()

    if SAVE_RESULTS:
        trace_volume.commit()

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.solution != solution.name or not trace.evaluation:
            continue
        status = trace.evaluation.status
        entry = {
            "status": status.value if hasattr(status, "value") else str(status),
            "solution": trace.solution,
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main(
    resume: bool = DEFAULT_RESUME_RESULTS,
    solution_name_suffix: str = "",
):
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    if solution_name_suffix:
        solution = solution.model_copy(update={"name": f"{solution.name}{solution_name_suffix}"})
    print(f"Loaded: {solution.name} ({solution.definition})")
    print(f"Using image: {OFFICIAL_DOCKER_IMAGE}")
    print(f"Dataset volume: {TRACE_VOLUME_NAME} -> {TRACE_SET_PATH}")
    print(f"Resume existing traces: {resume}")
    print("\nEquivalent flashinfer-bench command:")
    print(build_equivalent_cli(solution, resume))

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution, resume)

    if not results:
        print("No results returned!")
        return

    print_results(results)
