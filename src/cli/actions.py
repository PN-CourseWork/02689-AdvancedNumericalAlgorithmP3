"""CLI actions - the actual work functions."""

import subprocess
from pathlib import Path

from .console import console, ok, fail, dim, header

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_cmd(
    cmd: list[str], timeout: int = 180, cwd: Path = REPO_ROOT
) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd)
    )


def fetch_mlflow():
    """Fetch artifacts from MLflow."""
    header("Fetching MLflow artifacts...")
    try:
        from utils import download_artifacts_with_naming, setup_mlflow_auth

        setup_mlflow_auth()

        fv_paths = download_artifacts_with_naming(
            "HPC-FV-Solver", REPO_ROOT / "data" / "FV-Solver"
        )
        ok(f"Downloaded {len(fv_paths)} FV-Solver files")

        spectral_paths = download_artifacts_with_naming(
            "HPC-Spectral-Chebyshev",
            REPO_ROOT / "data" / "Spectral-Solver" / "Chebyshev",
        )
        ok(f"Downloaded {len(spectral_paths)} Spectral files")
    except Exception as e:
        fail(f"Failed: {e}")


def run_scripts(pattern: str):
    """Run scripts matching pattern (compute/plot)."""
    experiments_dir = REPO_ROOT / "Experiments"
    if not experiments_dir.exists():
        dim("No Experiments directory found")
        return

    scripts = sorted(
        [p for p in experiments_dir.rglob("*.py") if p.is_file() and pattern in p.name]
    )

    if not scripts:
        dim(f"No {pattern} scripts found")
        return

    header(f"Running {len(scripts)} {pattern} scripts...")

    for script in scripts:
        name = script.relative_to(REPO_ROOT)
        console.print(f"\n[bold cyan]▶ {name}[/bold cyan]")
        try:
            # Stream output directly to terminal
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                cwd=str(REPO_ROOT),
                timeout=180,
            )
            ok(f"{name}") if result.returncode == 0 else fail(
                f"{name} (exit {result.returncode})"
            )
        except subprocess.TimeoutExpired:
            fail(f"{name} (timeout)")
        except Exception as e:
            fail(f"{name} ({e})")


def build_docs():
    """Build Sphinx documentation."""
    header("Building documentation...")
    try:
        result = run_cmd(
            [
                "uv",
                "run",
                "sphinx-build",
                "-M",
                "html",
                str(REPO_ROOT / "docs" / "source"),
                str(REPO_ROOT / "docs" / "build"),
            ],
            timeout=300,
        )

        if result.returncode == 0:
            ok("Documentation built")
            dim(f"Open: {REPO_ROOT / 'docs' / 'build' / 'html' / 'index.html'}")
        else:
            fail(f"Build failed (exit {result.returncode})")
    except Exception as e:
        fail(f"Build failed: {e}")


def clean_all():
    """Clean generated files and caches."""
    import shutil

    header("Cleaning...")
    targets = [
        "docs/build",
        "docs/source/example_gallery",
        "docs/source/generated",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    ]

    count = 0
    for target in targets:
        path = REPO_ROOT / target
        if path.exists():
            shutil.rmtree(path)
            count += 1

    for pycache in REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache)
        count += 1

    # Clean data/ but keep README.md
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.name not in ("README.md", ".gitkeep"):
                shutil.rmtree(item) if item.is_dir() else item.unlink()
                count += 1

    ok(f"Cleaned {count} items") if count else dim("Nothing to clean")


def ruff_check():
    """Run ruff linter."""
    header("Running ruff check...")
    result = run_cmd(["uv", "run", "ruff", "check", "."], timeout=60)
    print(result.stdout)
    ok("No issues") if result.returncode == 0 else fail(
        f"Found issues (exit {result.returncode})"
    )


def ruff_format():
    """Run ruff formatter."""
    header("Running ruff format...")
    result = run_cmd(["uv", "run", "ruff", "format", "."], timeout=60)
    print(result.stdout)
    ok("Formatted") if result.returncode == 0 else fail(
        f"Failed (exit {result.returncode})"
    )


def hpc_submit(experiment: str = "all", dry_run: bool = True):
    """Submit HPC jobs from YAML configs."""
    from .hpc import (
        discover_experiments,
        generate_pack,
        get_experiment_name,
        load_config,
        generate_jobs,
    )

    experiments = discover_experiments()
    if not experiments:
        fail("No experiments with jobs.yaml found")
        return

    # Filter experiments
    if experiment != "all":
        experiments = [
            e for e in experiments if experiment.lower() in e.parent.name.lower()
        ]

    if not experiments:
        fail(f"No matching experiment: {experiment}")
        return

    for yaml_path in experiments:
        name = get_experiment_name(yaml_path)
        console.print(f"\n[bold]{name}:[/bold]")

        try:
            config = load_config(yaml_path)
            jobs = generate_jobs(config)
            pack_content = generate_pack(yaml_path)

            dim(f"Generated {len(jobs)} jobs from {yaml_path.name}")

            if dry_run:
                for job in jobs:
                    console.print(f"    [cyan]{job['name']}[/cyan]")
            else:
                # Ensure logs directory exists
                logs_dir = REPO_ROOT / "logs"
                logs_dir.mkdir(exist_ok=True)

                # Write and submit pack file
                pack_file = yaml_path.parent / "jobs.pack"
                pack_file.write_text(pack_content)

                result = subprocess.run(
                    ["bsub", "-pack", str(pack_file)],
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT),
                )
                if result.returncode == 0:
                    ok(f"Submitted {len(jobs)} jobs")
                    if result.stdout.strip():
                        dim(result.stdout.strip())
                else:
                    fail(f"Submission failed: {result.stderr}")

        except Exception as e:
            fail(f"Error: {e}")
