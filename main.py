#!/usr/bin/env python3
"""Main script to run all examples."""

import argparse
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get repository root directory.

    Returns the repository root by detecting the presence of pyproject.toml.
    Works from any subdirectory of the repository.

    Returns
    -------
    Path
        Absolute path to the repository root

    """
    # Start from this script's location
    current = Path(__file__).resolve().parent

    # Walk up until we find pyproject.toml (marks repo root)
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume script is in repo root
    return current


# Get repo root once at module level
REPO_ROOT = get_repo_root()


def discover_scripts():
    """Find all Python scripts in Experiments/ directory."""
    experiments_dir = REPO_ROOT / "Experiments"

    if not experiments_dir.exists():
        return [], []

    scripts = [
        p
        for p in experiments_dir.rglob("*.py")
        if p.is_file() and p.name != "__init__.py"
    ]

    compute_scripts = sorted([s for s in scripts if "compute" in s.name])
    plot_scripts = sorted([s for s in scripts if "plot" in s.name])

    return compute_scripts, plot_scripts


def run_scripts(scripts, console=None):
    """Run scripts sequentially and report results."""
    if not scripts:
        if console:
            console.print("  No scripts to run", style="dim")
        else:
            print("  No scripts to run")
        return

    msg = f"\nRunning {len(scripts)} scripts...\n"
    if console:
        console.print(msg)
    else:
        print(msg)

    success_count = 0
    fail_count = 0

    for script in scripts:
        # Display path relative to repo root
        display_path = script.relative_to(REPO_ROOT)

        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(REPO_ROOT),  # Run from repo root
            )

            if result.returncode == 0:
                if console:
                    console.print(f"  [green]✓[/green] {display_path}")
                else:
                    print(f"  ✓ {display_path}")
                success_count += 1
            else:
                if console:
                    console.print(
                        f"  [red]✗[/red] {display_path} (exit {result.returncode})"
                    )
                    if result.stderr:
                        console.print(f"    [dim]{result.stderr[:200]}[/dim]")
                else:
                    print(f"  ✗ {display_path} (exit {result.returncode})")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}")
                fail_count += 1

        except subprocess.TimeoutExpired:
            if console:
                console.print(f"  [red]✗[/red] {display_path} (timeout)")
            else:
                print(f"  ✗ {display_path} (timeout)")
            fail_count += 1
        except Exception as e:
            if console:
                console.print(f"  [red]✗[/red] {display_path} ({e})")
            else:
                print(f"  ✗ {display_path} ({e})")
            fail_count += 1

    summary = f"\n  Summary: {success_count} succeeded, {fail_count} failed\n"
    if console:
        console.print(summary)
    else:
        print(summary)


def build_docs(console=None):
    """Build Sphinx documentation."""
    docs_dir = REPO_ROOT / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    msg = "\nBuilding Sphinx documentation..."
    if console:
        console.print(msg)
    else:
        print(msg)

    if not source_dir.exists():
        err = f"  Error: Documentation source directory not found: {source_dir}"
        if console:
            console.print(f"  [red]{err}[/red]")
        else:
            print(err)
        return False

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "sphinx-build",
                "-M",
                "html",
                str(source_dir),
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            if console:
                console.print("  [green]✓[/green] Documentation built successfully")
                console.print(f"  → Open: {build_dir / 'html' / 'index.html'}\n")
            else:
                print("  ✓ Documentation built successfully")
                print(f"  → Open: {build_dir / 'html' / 'index.html'}\n")
            return True
        else:
            if console:
                console.print(
                    f"  [red]✗[/red] Documentation build failed (exit {result.returncode})"
                )
                if result.stderr:
                    console.print(f"    [dim]{result.stderr[:500]}[/dim]")
            else:
                print(f"  ✗ Documentation build failed (exit {result.returncode})")
                if result.stderr:
                    print(f"    Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        if console:
            console.print("  [red]✗[/red] Documentation build timed out")
        else:
            print("  ✗ Documentation build timed out")
        return False
    except FileNotFoundError:
        if console:
            console.print(
                "  [red]✗[/red] sphinx-build not found. Install with: uv sync"
            )
        else:
            print("  ✗ sphinx-build not found. Install with: uv sync")
        return False
    except Exception as e:
        if console:
            console.print(f"  [red]✗[/red] Documentation build failed: {e}")
        else:
            print(f"  ✗ Documentation build failed: {e}")
        return False


def clean_docs(console=None):
    """Clean built Sphinx documentation."""
    import shutil

    build_dir = REPO_ROOT / "docs" / "build"

    msg = "\nCleaning Sphinx documentation..."
    if console:
        console.print(msg)
    else:
        print(msg)

    if not build_dir.exists():
        if console:
            console.print(f"  [dim]No build directory found at {build_dir}[/dim]")
        else:
            print(f"  No build directory found at {build_dir}")
        return

    try:
        shutil.rmtree(build_dir)
        if console:
            console.print(
                f"  [green]✓[/green] Cleaned {build_dir.relative_to(REPO_ROOT)}\n"
            )
        else:
            print(f"  ✓ Cleaned {build_dir.relative_to(REPO_ROOT)}\n")
    except Exception as e:
        if console:
            console.print(f"  [red]✗[/red] Failed to clean documentation: {e}\n")
        else:
            print(f"  ✗ Failed to clean documentation: {e}\n")


def clean_all(console=None):
    """Clean all generated files and caches."""
    import shutil

    msg = "\nCleaning all generated files and caches..."
    if console:
        console.print(msg)
    else:
        print(msg)

    cleaned = []
    failed = []

    # List of paths to clean (relative to repo root)
    clean_targets = [
        REPO_ROOT / "docs" / "build",
        REPO_ROOT / "docs" / "source" / "example_gallery",
        REPO_ROOT / "docs" / "source" / "generated",
        REPO_ROOT / "build",
        REPO_ROOT / "dist",
        REPO_ROOT / ".pytest_cache",
        REPO_ROOT / ".ruff_cache",
        REPO_ROOT / ".mypy_cache",
    ]

    # Clean directories
    for target_path in clean_targets:
        if target_path.exists():
            try:
                shutil.rmtree(target_path)
                cleaned.append(str(target_path.relative_to(REPO_ROOT)))
            except Exception as e:
                failed.append(f"{target_path.relative_to(REPO_ROOT)}: {e}")

    # Clean __pycache__ directories
    for pycache in REPO_ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            cleaned.append(str(pycache.relative_to(REPO_ROOT)))
        except Exception as e:
            failed.append(f"{pycache.relative_to(REPO_ROOT)}: {e}")

    # Clean .pyc files
    for pyc in REPO_ROOT.rglob("*.pyc"):
        try:
            pyc.unlink()
            cleaned.append(str(pyc.relative_to(REPO_ROOT)))
        except Exception as e:
            failed.append(f"{pyc.relative_to(REPO_ROOT)}: {e}")

    # Clean data directory (but keep README.md)
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.name != "README.md" and item.name != ".gitkeep":
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    cleaned.append(str(item.relative_to(REPO_ROOT)))
                except Exception as e:
                    failed.append(f"{item.relative_to(REPO_ROOT)}: {e}")

    # Print results
    if cleaned:
        if console:
            console.print(f"  [green]✓[/green] Cleaned {len(cleaned)} items")
        else:
            print(f"  ✓ Cleaned {len(cleaned)} items")
    if failed:
        if console:
            console.print(f"  [red]✗[/red] Failed to clean {len(failed)} items:")
            for fail in failed[:5]:
                console.print(f"    [dim]- {fail}[/dim]")
        else:
            print(f"  ✗ Failed to clean {len(failed)} items:")
            for fail in failed[:5]:
                print(f"    - {fail}")
    if not cleaned and not failed:
        if console:
            console.print("  [dim]Nothing to clean[/dim]")
        else:
            print("  Nothing to clean")
    print()


def hpc_submit(solver: str, dry_run: bool = False, console=None):
    """Generate and submit HPC job pack.

    Parameters
    ----------
    solver : str
        Which solver to submit: 'spectral', 'fv', or 'all'
    dry_run : bool
        If True, only print the jobs without submitting
    """
    solvers = []
    if solver in ("spectral", "all"):
        solvers.append(("Spectral-Solver", REPO_ROOT / "Experiments" / "Spectral-Solver"))
    if solver in ("fv", "all"):
        solvers.append(("FV-Solver", REPO_ROOT / "Experiments" / "FV-Solver"))

    for name, solver_dir in solvers:
        generator = solver_dir / "generate_pack.sh"
        pack_file = solver_dir / "jobs.pack"

        if not generator.exists():
            if console:
                console.print(f"  [red]✗[/red] {name}: generate_pack.sh not found")
            else:
                print(f"  ✗ {name}: generate_pack.sh not found")
            continue

        if console:
            console.print(f"\n[bold]{name}:[/bold]")
        else:
            print(f"\n{name}:")

        # Generate pack file
        try:
            result = subprocess.run(
                ["bash", str(generator)],
                capture_output=True,
                text=True,
                cwd=str(solver_dir),
            )
            jobs = result.stdout.strip()

            if not jobs:
                if console:
                    console.print("  [red]✗[/red] No jobs generated")
                else:
                    print("  ✗ No jobs generated")
                continue

            job_count = len(jobs.split("\n"))
            if console:
                console.print(f"  Generated {job_count} jobs")
            else:
                print(f"  Generated {job_count} jobs")

            if dry_run:
                if console:
                    console.print("\n  [dim]Jobs to submit:[/dim]")
                else:
                    print("\n  Jobs to submit:")
                for line in jobs.split("\n"):
                    # Extract job name from -J flag
                    parts = line.split()
                    job_name = parts[1] if len(parts) > 1 else "unknown"
                    if console:
                        console.print(f"    [cyan]- {job_name}[/cyan]")
                    else:
                        print(f"    - {job_name}")
            else:
                # Write pack file
                pack_file.write_text(jobs)

                # Submit to LSF from repo root
                result = subprocess.run(
                    ["bsub", "-pack", str(pack_file)],
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT),
                )

                if result.returncode == 0:
                    if console:
                        console.print(f"  [green]✓[/green] Submitted {job_count} jobs")
                        console.print(f"  {result.stdout.strip()}")
                    else:
                        print(f"  ✓ Submitted {job_count} jobs")
                        print(f"  {result.stdout.strip()}")
                else:
                    if console:
                        console.print(
                            f"  [red]✗[/red] Submission failed: {result.stderr}"
                        )
                    else:
                        print(f"  ✗ Submission failed: {result.stderr}")

        except FileNotFoundError:
            if console:
                console.print("  [red]✗[/red] bsub not found (are you on HPC?)")
            else:
                print("  ✗ bsub not found (are you on HPC?)")
        except Exception as e:
            if console:
                console.print(f"  [red]✗[/red] Error: {e}")
            else:
                print(f"  ✗ Error: {e}")


def fetch_mlflow(console=None):
    """Fetch artifacts from MLflow for all converged runs."""
    msg = "\nFetching MLflow artifacts..."
    if console:
        console.print(msg)
    else:
        print(msg)

    try:
        from utils import download_artifacts_with_naming, setup_mlflow_auth

        setup_mlflow_auth()

        fv_dir = REPO_ROOT / "data" / "FV-Solver"
        if console:
            console.print("\n[bold]Finite Volume:[/bold]")
        else:
            print("\nFinite Volume:")
        fv_paths = download_artifacts_with_naming("HPC-FV-Solver", fv_dir)
        if console:
            console.print(
                f"  [green]✓[/green] Downloaded {len(fv_paths)} files to data/FV-Solver/"
            )
        else:
            print(f"  ✓ Downloaded {len(fv_paths)} files to data/FV-Solver/")

        spectral_dir = REPO_ROOT / "data" / "Spectral-Solver" / "Chebyshev"
        if console:
            console.print("\n[bold]Spectral:[/bold]")
        else:
            print("\nSpectral:")
        spectral_paths = download_artifacts_with_naming(
            "HPC-Spectral-Chebyshev", spectral_dir
        )
        if console:
            console.print(
                f"  [green]✓[/green] Downloaded {len(spectral_paths)} files to data/Spectral-Solver/"
            )
        else:
            print(f"  ✓ Downloaded {len(spectral_paths)} files to data/Spectral-Solver/")

        print()

    except ImportError as e:
        if console:
            console.print(f"  [red]✗[/red] Missing dependency: {e}")
            console.print("    [dim]Install with: uv sync[/dim]")
        else:
            print(f"  ✗ Missing dependency: {e}")
            print("    Install with: uv sync")
    except Exception as e:
        if console:
            console.print(f"  [red]✗[/red] Failed to fetch: {e}\n")
        else:
            print(f"  ✗ Failed to fetch: {e}\n")


def ruff_check(console=None):
    """Run ruff linter."""
    msg = "\nRunning ruff check..."
    if console:
        console.print(msg)
    else:
        print(msg)

    try:
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "."],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            if console:
                console.print("  [green]✓[/green] No issues found\n")
            else:
                print("  ✓ No issues found\n")
            return True
        else:
            if console:
                console.print(
                    f"  [red]✗[/red] Found issues (exit code {result.returncode})\n"
                )
            else:
                print(f"  ✗ Found issues (exit code {result.returncode})\n")
            return False

    except FileNotFoundError:
        if console:
            console.print("  [red]✗[/red] ruff not found. Install with: uv sync\n")
        else:
            print("  ✗ ruff not found. Install with: uv sync\n")
        return False
    except subprocess.TimeoutExpired:
        if console:
            console.print("  [red]✗[/red] ruff check timed out\n")
        else:
            print("  ✗ ruff check timed out\n")
        return False
    except Exception as e:
        if console:
            console.print(f"  [red]✗[/red] ruff check failed: {e}\n")
        else:
            print(f"  ✗ ruff check failed: {e}\n")
        return False


def ruff_format(console=None):
    """Run ruff formatter."""
    msg = "\nRunning ruff format..."
    if console:
        console.print(msg)
    else:
        print(msg)

    try:
        result = subprocess.run(
            ["uv", "run", "ruff", "format", "."],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            if console:
                console.print("  [green]✓[/green] Code formatted successfully\n")
            else:
                print("  ✓ Code formatted successfully\n")
            return True
        else:
            if console:
                console.print(
                    f"  [red]✗[/red] Formatting failed (exit code {result.returncode})\n"
                )
            else:
                print(f"  ✗ Formatting failed (exit code {result.returncode})\n")
            return False

    except FileNotFoundError:
        if console:
            console.print("  [red]✗[/red] ruff not found. Install with: uv sync\n")
        else:
            print("  ✗ ruff not found. Install with: uv sync\n")
        return False
    except subprocess.TimeoutExpired:
        if console:
            console.print("  [red]✗[/red] ruff format timed out\n")
        else:
            print("  ✗ ruff format timed out\n")
        return False
    except Exception as e:
        if console:
            console.print(f"  [red]✗[/red] ruff format failed: {e}\n")
        else:
            print(f"  ✗ ruff format failed: {e}\n")
        return False


def interactive_menu():
    """Run interactive CLI menu."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    console = Console()

    # Menu options
    menu_items = [
        ("1", "fetch", "Fetch MLflow artifacts", lambda: fetch_mlflow(console)),
        ("2", "compute", "Run compute scripts", lambda: run_scripts(discover_scripts()[0], console)),
        ("3", "plot", "Run plot scripts", lambda: run_scripts(discover_scripts()[1], console)),
        ("4", "docs", "Build documentation", lambda: build_docs(console)),
        ("5", "lint", "Run ruff linter", lambda: ruff_check(console)),
        ("6", "format", "Format code with ruff", lambda: ruff_format(console)),
        ("7", "clean", "Clean all caches", lambda: clean_all(console)),
        ("8", "clean-docs", "Clean documentation", lambda: clean_docs(console)),
        ("9", "hpc", "Submit HPC jobs", None),  # Special handling
        ("q", "quit", "Exit", None),
    ]

    while True:
        console.clear()

        # Header
        console.print(
            Panel.fit(
                "[bold cyan]ANA-P3 Project Manager[/bold cyan]\n"
                "[dim]Advanced Numerical Algorithms - Project 3[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        # Build menu table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan", width=4)
        table.add_column("Command", style="bold", width=12)
        table.add_column("Description", style="dim")

        for key, cmd, desc, _ in menu_items:
            table.add_row(f"[{key}]", cmd, desc)

        console.print(table)
        console.print()

        # Get user choice
        choice = Prompt.ask(
            "[bold]Select an option[/bold]",
            choices=[item[0] for item in menu_items],
            show_choices=False,
        )

        if choice == "q":
            console.print("\n[dim]Goodbye![/dim]\n")
            break

        if choice == "9":
            # HPC submenu
            console.print()
            hpc_choice = Prompt.ask(
                "  [bold]Which solver?[/bold]",
                choices=["spectral", "fv", "all", "cancel"],
                default="all",
            )
            if hpc_choice != "cancel":
                dry_run = Prompt.ask(
                    "  [bold]Dry run?[/bold]",
                    choices=["yes", "no"],
                    default="yes",
                ) == "yes"
                hpc_submit(hpc_choice, dry_run=dry_run, console=console)
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
            continue

        # Find and execute the action
        for key, _, _, action in menu_items:
            if key == choice and action:
                console.print()
                action()
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run example scripts and manage documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              Interactive mode
  python main.py --compute                    Run data generation scripts
  python main.py --plot                       Run plotting scripts
  python main.py --fetch                      Fetch artifacts from MLflow
  python main.py --build-docs                 Build Sphinx HTML documentation
  python main.py --clean-docs                 Clean built documentation
  python main.py --clean-all                  Clean all generated files and caches
  python main.py --lint                       Check code with ruff
  python main.py --format                     Format code with ruff
  python main.py --compute --plot             Run all example scripts
  python main.py --hpc spectral               Submit spectral solver jobs to HPC
  python main.py --hpc all --dry-run          Preview all HPC jobs without submitting
        """,
    )

    parser.add_argument(
        "--compute", action="store_true", help="Run data generation (compute) scripts"
    )
    parser.add_argument("--plot", action="store_true", help="Run plotting scripts")
    parser.add_argument(
        "--build-docs", action="store_true", help="Build Sphinx HTML documentation"
    )
    parser.add_argument(
        "--clean-docs", action="store_true", help="Clean built Sphinx documentation"
    )
    parser.add_argument(
        "--clean-all", action="store_true", help="Clean all generated files and caches"
    )
    parser.add_argument("--lint", action="store_true", help="Run ruff linter")
    parser.add_argument("--format", action="store_true", help="Run ruff formatter")
    parser.add_argument(
        "--hpc",
        choices=["spectral", "fv", "all"],
        help="Submit jobs to HPC (spectral, fv, or all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview HPC jobs without submitting",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch artifacts from MLflow for all converged runs",
    )

    args = parser.parse_args()

    # Check if any arguments were provided
    has_args = any([
        args.compute,
        args.plot,
        args.build_docs,
        args.clean_docs,
        args.clean_all,
        args.lint,
        args.format,
        args.hpc,
        args.fetch,
    ])

    # If no arguments, run interactive mode
    if not has_args:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
        return

    # Handle cleaning commands
    if args.clean_all:
        clean_all()

    if args.clean_docs:
        clean_docs()

    # Handle code quality commands
    if args.lint:
        ruff_check()

    if args.format:
        ruff_format()

    # Handle HPC submission
    if args.hpc:
        hpc_submit(args.hpc, dry_run=args.dry_run)

    # Handle MLflow fetch
    if args.fetch:
        fetch_mlflow()

    # Handle documentation commands
    if args.build_docs:
        build_docs()

    # Handle example scripts
    if args.compute or args.plot:
        compute_scripts, plot_scripts = discover_scripts()
        print(
            f"\nFound {len(compute_scripts)} compute scripts and {len(plot_scripts)} plot scripts"
        )

        if args.compute:
            run_scripts(compute_scripts)
        if args.plot:
            run_scripts(plot_scripts)


if __name__ == "__main__":
    main()
