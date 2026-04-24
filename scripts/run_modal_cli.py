"""
FlashInfer-Bench Modal runner that shells out to the official CLI.

This wraps:

flashinfer-bench run \
  --local ./contest-dataset \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 \
  --atol 1 --rtol 0.3 --required-matched-ratio 0.9
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet

app = modal.App("flashinfer-bench-cli")

OFFICIAL_DOCKER_IMAGE = "flashinfer/flashinfer-ci-cu132:20260401-2c675fb"
TRACE_VOLUME_NAME = "flashinfer-trace"
TRACE_SET_PATH = "/data"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_RESUME_RESULTS = True
LOG_LEVEL = "INFO"

DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
ATOL = 1.0
RTOL = 0.3
REQUIRED_MATCHED_RATIO = 0.9

trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(OFFICIAL_DOCKER_IMAGE)
    .env({"PYTHONUNBUFFERED": "1"})
    .run_commands("python -m pip install -U flashinfer-bench")
)


def build_cli_args(solution: Solution, resume: bool) -> list[str]:
    args = [
        "flashinfer-bench",
        "run",
        "--local",
        TRACE_SET_PATH,
        "--definitions",
        DEFINITION,
        "--solutions",
        solution.name,
        "--save-results",
        "--use-isolated-runner",
        "--log-level",
        LOG_LEVEL,
        "--timeout",
        str(DEFAULT_TIMEOUT_SECONDS),
        "--atol",
        f"{ATOL:g}",
        "--rtol",
        f"{RTOL:g}",
        "--required-matched-ratio",
        f"{REQUIRED_MATCHED_RATIO:g}",
    ]
    if resume:
        args.append("--resume")
    return args


def format_cli(args: list[str]) -> str:
    chunks = []
    i = 0
    while i < len(args):
        token = args[i]
        if i == 0:
            chunks.append(token)
            i += 1
            continue
        if token.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            chunks.append(f"  {token} {shlex.quote(args[i + 1])}")
            i += 2
            continue
        chunks.append(f"  {shlex.quote(token)}")
        i += 1
    return " \\\n".join(chunks)


def trace_solution_path(solution: Solution, trace_set: TraceSet) -> Path:
    definition = trace_set.definitions[solution.definition]
    return (
        Path(TRACE_SET_PATH)
        / "solutions"
        / solution.author
        / definition.op_type
        / solution.definition
        / f"{solution.name}.json"
    )


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark_cli(solution_json: str, resume: bool = DEFAULT_RESUME_RESULTS) -> dict:
    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition != DEFINITION:
        raise ValueError(f"Expected definition '{DEFINITION}', got '{solution.definition}'")
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    solution_path = trace_solution_path(solution, trace_set)
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    solution_path.write_text(solution_json, encoding="utf-8")
    trace_volume.commit()

    cli_args = build_cli_args(solution, resume)
    proc = subprocess.run(cli_args, text=True, capture_output=True, check=False)

    trace_volume.commit()
    result = {
        "command": format_cli(cli_args),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "solution_path": str(solution_path),
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    resume: bool = DEFAULT_RESUME_RESULTS,
    solution_name_suffix: str = "",
):
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    if solution_name_suffix:
        solution = solution.model_copy(update={"name": f"{solution.name}{solution_name_suffix}"})

    cli_args = build_cli_args(solution, resume)
    print(f"Loaded: {solution.name} ({solution.definition})")
    print(f"Using image: {OFFICIAL_DOCKER_IMAGE}")
    print(f"Dataset volume: {TRACE_VOLUME_NAME} -> {TRACE_SET_PATH}")
    print(f"Resume existing traces: {resume}")
    print("\nCLI command to run in Modal:")
    print(format_cli(cli_args))

    print("\nRunning benchmark on Modal B200 via CLI...")
    result = run_benchmark_cli.remote(solution.model_dump_json(), resume)

    print("\nRemote stdout:")
    print(result["stdout"] or "(empty)")
    if result["stderr"]:
        print("\nRemote stderr:")
        print(result["stderr"])
