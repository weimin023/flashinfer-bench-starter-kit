"""Pack solution source files into `solution.json`."""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec, Solution, SourceFile


VALID_SOURCE_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp"}


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def resolve_source_dir(language: str, source_dir_name: str | None) -> Path:
    """Resolve the configured source directory."""
    if source_dir_name is None:
        if language == "triton":
            return PROJECT_ROOT / "solution" / "triton"
        if language in {"cuda", "python"}:
            return PROJECT_ROOT / "solution" / "cuda"
        raise ValueError(f"Unsupported language: {language}")
    return PROJECT_ROOT / "solution" / source_dir_name


def build_spec_from_config(build_config: dict) -> BuildSpec:
    """Build a BuildSpec from config.toml."""
    return BuildSpec(
        language=build_config["language"],
        target_hardware=["cuda"],
        entry_point=build_config["entry_point"],
        destination_passing_style=build_config.get("destination_passing_style", True),
        binding=build_config.get("binding"),
    )


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    spec = build_spec_from_config(build_config)
    source_dir = resolve_source_dir(spec.language, build_config.get("source_dir"))

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    sources = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") for part in file_path.relative_to(source_dir).parts):
            continue
        if file_path.suffix.lower() not in VALID_SOURCE_EXTENSIONS:
            continue
        rel_path = file_path.relative_to(source_dir).as_posix()
        sources.append(SourceFile(path=rel_path, content=file_path.read_text(encoding="utf-8")))

    if not sources:
        raise ValueError(f"No source files found in directory: {source_dir}")

    solution = Solution(
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
        spec=spec,
        sources=sources,
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {spec.language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
