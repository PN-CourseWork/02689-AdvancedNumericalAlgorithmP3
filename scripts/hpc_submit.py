#!/usr/bin/env python
"""
HPC Job Array Submission Script.

Submits Hydra experiment sweeps as LSF job arrays on DTU HPC.

Usage:
    uv run python scripts/hpc_submit.py +experiment/validation/ghia=fv --dry-run
    uv run python scripts/hpc_submit.py +experiment/validation/ghia=fv
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


def parse_sweep_params(experiment_path: str) -> dict[str, list]:
    """Parse sweep parameters from experiment YAML."""
    if experiment_path.startswith("+"):
        experiment_path = experiment_path[1:]

    if "=" in experiment_path:
        group, name = experiment_path.rsplit("=", 1)
        yaml_path = Path("conf") / f"{group}/{name}.yaml"
    else:
        yaml_path = Path("conf") / f"{experiment_path}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {yaml_path}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    sweep_params = config.get("hydra", {}).get("sweeper", {}).get("params", {})
    if not sweep_params:
        raise ValueError(f"No sweep parameters found in {yaml_path}")

    parsed = {}
    for key, value in sweep_params.items():
        if isinstance(value, str):
            parsed[key] = [v.strip() for v in value.split(",")]
        elif isinstance(value, list):
            parsed[key] = [str(v) for v in value]
        else:
            parsed[key] = [str(value)]

    return parsed


def generate_combinations(sweep_params: dict[str, list]) -> list[dict[str, str]]:
    """Generate all parameter combinations."""
    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def get_command_for_index(experiment: str, combinations: list[dict], index: int) -> str:
    """Get the command for a specific job index (1-indexed)."""
    combo = combinations[index - 1]  # Convert to 0-indexed
    overrides = " ".join(f"{k}={v}" for k, v in combo.items())
    return f"uv run python main.py +{experiment} {overrides} mlflow=coolify"


def main():
    parser = argparse.ArgumentParser(description="Submit HPC job array")
    parser.add_argument("experiment", help="Experiment (e.g., +experiment/validation/ghia=fv)")
    parser.add_argument("--queue", "-q", default="hpc", help="LSF queue (default: hpc)")
    parser.add_argument("--time", "-W", default="2:00", help="Wall time (default: 2:00)")
    parser.add_argument("--cores", "-n", type=int, default=4, help="Cores per job (default: 4)")
    parser.add_argument("--mem", default="6GB", help="Memory per core (default: 4GB)")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without submitting")
    parser.add_argument("--test-index", type=int, help="Test: show command for specific index")

    args = parser.parse_args()

    # Parse and generate combinations
    experiment = args.experiment.lstrip("+")
    sweep_params = parse_sweep_params(args.experiment)
    combinations = generate_combinations(sweep_params)

    print(f"Experiment: {args.experiment}")
    print(f"Sweep: {sweep_params}")
    print(f"Jobs: {len(combinations)}")

    # Test mode: show command for specific index
    if args.test_index:
        if args.test_index < 1 or args.test_index > len(combinations):
            print(f"Error: index must be 1-{len(combinations)}", file=sys.stderr)
            sys.exit(1)
        cmd = get_command_for_index(experiment, combinations, args.test_index)
        print(f"\n[{args.test_index}] {combinations[args.test_index - 1]}")
        print(f"Command: {cmd}")
        return

    # Show all combinations
    for i, combo in enumerate(combinations, 1):
        print(f"  [{i}] {combo}")

    if args.dry_run:
        print("\n--- Commands (dry run) ---")
        for i in range(1, len(combinations) + 1):
            print(f"[{i}] {get_command_for_index(experiment, combinations, i)}")
        return

    # Submit job array
    job_name = experiment.replace("/", "_").replace("=", "_")
    n_jobs = len(combinations)

    # Build bash arrays for parameter mapping
    # Replace dots with underscores for valid bash variable names
    param_keys = list(combinations[0].keys())
    array_defs = []
    for key in param_keys:
        vals = " ".join(c[key] for c in combinations)
        bash_var = key.upper().replace(".", "_")
        array_defs.append(f'{bash_var}=({vals})')

    # Build overrides: use bash var name but original key for hydra
    overrides_parts = []
    for k in param_keys:
        bash_var = k.upper().replace(".", "_")
        overrides_parts.append(f'{k}=${{{bash_var}[$I]}}')
    overrides = " ".join(overrides_parts)

    script = f"""#!/bin/bash
mkdir -p logs
{chr(10).join(array_defs)}
I=$((LSB_JOBINDEX - 1))
uv run python main.py +{experiment} {overrides} mlflow=coolify
"""

    bsub_cmd = [
        "bsub",
        "-J", f"{job_name}[1-{n_jobs}]",
        "-q", args.queue,
        "-W", args.time,
        "-n", str(args.cores),
        "-R", f"rusage[mem={args.mem}]",
        "-R", "span[hosts=1]",
        "-o", "logs/%J_%I.out",
        "-e", "logs/%J_%I.err",
    ]

    print(f"\nSubmitting: {' '.join(bsub_cmd)}")
    result = subprocess.run(bsub_cmd, input=script, text=True, capture_output=True)

    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
