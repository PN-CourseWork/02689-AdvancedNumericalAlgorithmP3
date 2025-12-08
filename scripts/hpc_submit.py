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
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


def create_parent_run(experiment: str, sweep_params: dict) -> str:
    """Create MLflow parent run before submitting HPC jobs.

    This avoids race conditions when multiple jobs start simultaneously.
    """
    import mlflow

    load_dotenv()

    # Load experiment config to get experiment_name and sweep_name
    if "=" in experiment:
        group, name = experiment.rsplit("=", 1)
        yaml_path = Path("conf") / f"{group}/{name}.yaml"
    else:
        yaml_path = Path("conf") / f"{experiment}.yaml"

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Load mlflow coolify config
    mlflow_config_path = Path("conf/mlflow/coolify.yaml")
    with open(mlflow_config_path) as f:
        mlflow_config = yaml.safe_load(f)

    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_config.get("tracking_uri"))
    mlflow.set_tracking_uri(tracking_uri)

    # Build experiment name
    experiment_name = config.get("experiment_name", "LDC-Validation")
    project_prefix = mlflow_config.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        full_experiment_name = f"{project_prefix}/{experiment_name}"
    else:
        full_experiment_name = experiment_name

    mlflow.set_experiment(full_experiment_name)

    # Create parent run
    sweep_name = config.get("sweep_name", experiment.replace("/", "_"))
    with mlflow.start_run(run_name=sweep_name) as parent_run:
        mlflow.set_tag("sweep", "parent")
        mlflow.log_dict({"sweep_params": sweep_params}, "sweep_config.yaml")

    return parent_run.info.run_id


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
    parser.add_argument("--time", "-W", default="6:00", help="Wall time (default: 1:00)")
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

    # Create parent run BEFORE submitting jobs to avoid race condition
    parent_run_id = create_parent_run(experiment, sweep_params)
    print(f"Created parent run: {parent_run_id}")

    script = f"""#!/bin/bash
mkdir -p logs
export MLFLOW_PARENT_RUN_ID={parent_run_id}
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
