#!/usr/bin/env python3
"""Project CLI - run with `uv run python main.py` for interactive mode."""

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Global console for rich output
console = Console()

# Repository root detection
REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: list[str], timeout: int = 180, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd))


def ok(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def fail(msg: str):
    console.print(f"  [red]✗[/red] {msg}")


def dim(msg: str):
    console.print(f"  [dim]{msg}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────


def fetch_mlflow():
    """Fetch artifacts from MLflow."""
    console.print("\n[bold]Fetching MLflow artifacts...[/bold]")
    try:
        from utils import download_artifacts_with_naming, setup_mlflow_auth

        setup_mlflow_auth()

        fv_paths = download_artifacts_with_naming("HPC-FV-Solver", REPO_ROOT / "data" / "FV-Solver")
        ok(f"Downloaded {len(fv_paths)} FV-Solver files")

        spectral_paths = download_artifacts_with_naming(
            "HPC-Spectral-Chebyshev", REPO_ROOT / "data" / "Spectral-Solver" / "Chebyshev"
        )
        ok(f"Downloaded {len(spectral_paths)} Spectral files")
    except Exception as e:
        fail(f"Failed: {e}")


def run_scripts(pattern: str):
    """Run scripts matching pattern (compute/plot)."""
    scripts = sorted([
        p for p in (REPO_ROOT / "Experiments").rglob("*.py")
        if p.is_file() and pattern in p.name
    ])

    if not scripts:
        dim(f"No {pattern} scripts found")
        return

    console.print(f"\n[bold]Running {len(scripts)} {pattern} scripts...[/bold]")

    for script in scripts:
        name = script.relative_to(REPO_ROOT)
        try:
            result = run_cmd(["uv", "run", "python", str(script)])
            ok(name) if result.returncode == 0 else fail(f"{name} (exit {result.returncode})")
        except subprocess.TimeoutExpired:
            fail(f"{name} (timeout)")
        except Exception as e:
            fail(f"{name} ({e})")


def build_docs():
    """Build Sphinx documentation."""
    console.print("\n[bold]Building documentation...[/bold]")
    try:
        result = run_cmd([
            "uv", "run", "sphinx-build", "-M", "html",
            str(REPO_ROOT / "docs" / "source"),
            str(REPO_ROOT / "docs" / "build"),
        ], timeout=300)

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

    console.print("\n[bold]Cleaning...[/bold]")
    targets = [
        "docs/build", "docs/source/example_gallery", "docs/source/generated",
        "build", "dist", ".pytest_cache", ".ruff_cache", ".mypy_cache",
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
    console.print("\n[bold]Running ruff check...[/bold]")
    result = run_cmd(["uv", "run", "ruff", "check", "."], timeout=60)
    print(result.stdout)
    ok("No issues") if result.returncode == 0 else fail(f"Found issues (exit {result.returncode})")


def ruff_format():
    """Run ruff formatter."""
    console.print("\n[bold]Running ruff format...[/bold]")
    result = run_cmd(["uv", "run", "ruff", "format", "."], timeout=60)
    print(result.stdout)
    ok("Formatted") if result.returncode == 0 else fail(f"Failed (exit {result.returncode})")


def hpc_submit(solver: str = "all", dry_run: bool = True):
    """Submit HPC jobs."""
    solvers = []
    if solver in ("spectral", "all"):
        solvers.append(("Spectral", REPO_ROOT / "Experiments" / "Spectral-Solver"))
    if solver in ("fv", "all"):
        solvers.append(("FV", REPO_ROOT / "Experiments" / "FV-Solver"))

    for name, path in solvers:
        gen = path / "generate_pack.sh"
        if not gen.exists():
            fail(f"{name}: generate_pack.sh not found")
            continue

        console.print(f"\n[bold]{name}:[/bold]")
        result = subprocess.run(["bash", str(gen)], capture_output=True, text=True, cwd=str(path))
        jobs = result.stdout.strip()

        if not jobs:
            fail("No jobs generated")
            continue

        job_count = len(jobs.split("\n"))
        dim(f"Generated {job_count} jobs")

        if dry_run:
            for line in jobs.split("\n"):
                parts = line.split()
                console.print(f"    [cyan]{parts[1] if len(parts) > 1 else 'unknown'}[/cyan]")
        else:
            pack_file = path / "jobs.pack"
            pack_file.write_text(jobs)
            result = subprocess.run(["bsub", "-pack", str(pack_file)], capture_output=True, text=True)
            ok(f"Submitted {job_count} jobs") if result.returncode == 0 else fail(result.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Menu
# ─────────────────────────────────────────────────────────────────────────────


def interactive():
    """Arrow-key navigation menu."""
    import questionary

    actions = [
        ("Fetch MLflow artifacts", fetch_mlflow),
        ("Run compute scripts", lambda: run_scripts("compute")),
        ("Run plot scripts", lambda: run_scripts("plot")),
        ("Build documentation", build_docs),
        ("Run ruff linter", ruff_check),
        ("Format code", ruff_format),
        ("Clean caches", clean_all),
        ("Submit HPC jobs", "hpc"),
        ("Exit", None),
    ]

    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]ANA-P3[/bold cyan] [dim]Advanced Numerical Algorithms[/dim]",
            border_style="cyan",
        ))
        console.print()

        choice = questionary.select(
            "Select action:",
            choices=[a[0] for a in actions],
            style=questionary.Style([("highlighted", "bold cyan"), ("pointer", "cyan")]),
        ).ask()

        if choice is None or choice == "Exit":
            console.print("[dim]Goodbye![/dim]\n")
            break

        for name, action in actions:
            if name == choice:
                if action == "hpc":
                    solver = questionary.select("Solver:", choices=["all", "spectral", "fv", "← Back"]).ask()
                    if solver and solver != "← Back":
                        dry_run = questionary.confirm("Dry run?", default=True).ask()
                        if dry_run is not None:
                            hpc_submit(solver, dry_run)
                            input("\nEnter to continue...")
                elif action:
                    action()
                    input("\nEnter to continue...")
                break


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ANA-P3 Project Manager")
    parser.add_argument("--fetch", action="store_true", help="Fetch MLflow artifacts")
    parser.add_argument("--compute", action="store_true", help="Run compute scripts")
    parser.add_argument("--plot", action="store_true", help="Run plot scripts")
    parser.add_argument("--build-docs", action="store_true", help="Build documentation")
    parser.add_argument("--clean", action="store_true", help="Clean caches")
    parser.add_argument("--lint", action="store_true", help="Run ruff linter")
    parser.add_argument("--format", action="store_true", help="Format with ruff")
    parser.add_argument("--hpc", choices=["spectral", "fv", "all"], help="Submit HPC jobs")
    parser.add_argument("--dry-run", action="store_true", help="HPC dry run")

    args = parser.parse_args()

    # No args = interactive mode
    if len(sys.argv) == 1:
        try:
            interactive()
        except KeyboardInterrupt:
            print("\n")
        return

    # CLI mode
    if args.clean:
        clean_all()
    if args.lint:
        ruff_check()
    if args.format:
        ruff_format()
    if args.fetch:
        fetch_mlflow()
    if args.build_docs:
        build_docs()
    if args.compute:
        run_scripts("compute")
    if args.plot:
        run_scripts("plot")
    if args.hpc:
        hpc_submit(args.hpc, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
