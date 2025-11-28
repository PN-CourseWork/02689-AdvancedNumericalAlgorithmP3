"""HPC job generation from YAML configs."""

import itertools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(yaml_path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def generate_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate job list from config with parameter sweep."""
    script = config["script"]
    lsf = config["lsf"]
    fixed_params = config.get("parameters", {})
    sweep = config.get("sweep", {})

    # Get sweep parameter names and values
    param_names = list(sweep.keys())
    param_values = list(sweep.values())

    # Generate cartesian product of all parameters
    if param_values:
        combinations = list(itertools.product(*param_values))
    else:
        combinations = [()]

    jobs = []
    for combo in combinations:
        sweep_params = dict(zip(param_names, combo))

        # Merge fixed params with sweep params (sweep overrides fixed)
        params = {**fixed_params, **sweep_params}

        # Build job name from sweep parameters only
        name_parts = [f"{k}{v}" for k, v in sweep_params.items()]
        job_name = "-".join(name_parts) if name_parts else "job"

        jobs.append(
            {
                "name": job_name,
                "script": script,
                "params": params,
                "lsf": lsf,
            }
        )

    return jobs


def job_to_pack_line(job: dict[str, Any]) -> str:
    """Convert job dict to LSF pack file line."""
    lsf = job["lsf"]
    name = job["name"]
    script = job["script"]
    params = job["params"]

    # Build LSF options
    parts = [
        f"-J {name}",
        f"-q {lsf['queue']}",
        f"-W {lsf['walltime']}",
        f"-n {lsf['cores']}",
        f'-R "rusage[mem={lsf["memory"]}]"',
        f"-o logs/{name}.out",
        f"-e logs/{name}.err",
    ]

    # Build command
    param_args = " ".join(f"--{k} {v}" for k, v in params.items())
    cmd = f"uv run python {script} {param_args}".strip()

    return " ".join(parts) + " " + cmd


def generate_pack(yaml_path: Path) -> str:
    """Generate full pack file content from YAML config."""
    config = load_config(yaml_path)
    jobs = generate_jobs(config)
    lines = [job_to_pack_line(job) for job in jobs]
    return "\n".join(lines)


def discover_experiments() -> list[Path]:
    """Find all experiments with jobs.yaml."""
    experiments_dir = REPO_ROOT / "Experiments"
    if not experiments_dir.exists():
        return []
    return sorted(experiments_dir.glob("*/jobs.yaml"))


def get_experiment_name(yaml_path: Path) -> str:
    """Extract experiment name from path."""
    return yaml_path.parent.name
