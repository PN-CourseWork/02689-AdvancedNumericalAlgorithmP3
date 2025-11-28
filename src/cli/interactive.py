"""Interactive CLI menu with arrow-key navigation."""

import questionary
from rich.panel import Panel

from .console import console
from .actions import (
    fetch_mlflow,
    run_scripts,
    build_docs,
    ruff_check,
    ruff_format,
    hpc_submit,
    hpc_status,
    hpc_monitor,
    hpc_kill,
    REPO_ROOT,
)

STYLE = questionary.Style([("highlighted", "bold cyan"), ("pointer", "cyan")])


def select(message: str, choices: list[str]) -> str | None:
    """Wrapped questionary select."""
    return questionary.select(message, choices=choices, style=STYLE).ask()


def confirm(message: str, default: bool = True) -> bool | None:
    """Wrapped questionary confirm."""
    return questionary.confirm(message, default=default).ask()


def wait():
    """Wait for Enter key."""
    input("\nEnter to continue...")


# ─────────────────────────────────────────────────────────────────────────────
# Submenus
# ─────────────────────────────────────────────────────────────────────────────


def menu_runner():
    """Runner submenu for compute/plot scripts."""
    while True:
        choice = select(
            "Runner:",
            [
                "Compute scripts",
                "Plot scripts",
                "← Back",
            ],
        )

        if choice == "Compute scripts":
            run_scripts("compute")
            wait()
        elif choice == "Plot scripts":
            run_scripts("plot")
            wait()
        else:
            break


def menu_clean():
    """Clean submenu."""
    import shutil

    while True:
        choice = select(
            "Clean:",
            [
                "Clean docs",
                "Clean data",
                "Clean caches",
                "Clean all",
                "← Back",
            ],
        )

        if choice == "Clean docs":
            build_dir = REPO_ROOT / "docs" / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
                console.print("  [green]✓[/green] Cleaned docs/build")
            else:
                console.print("  [dim]Nothing to clean[/dim]")
            wait()

        elif choice == "Clean data":
            data_dir = REPO_ROOT / "data"
            count = 0
            if data_dir.exists():
                for item in data_dir.iterdir():
                    if item.name not in ("README.md", ".gitkeep"):
                        shutil.rmtree(item) if item.is_dir() else item.unlink()
                        count += 1
            console.print(
                f"  [green]✓[/green] Cleaned {count} items"
            ) if count else console.print("  [dim]Nothing to clean[/dim]")
            wait()

        elif choice == "Clean caches":
            targets = [".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"]
            count = 0
            for t in targets:
                path = REPO_ROOT / t
                if path.exists():
                    shutil.rmtree(path)
                    count += 1
            for pycache in REPO_ROOT.rglob("__pycache__"):
                shutil.rmtree(pycache)
                count += 1
            console.print(
                f"  [green]✓[/green] Cleaned {count} items"
            ) if count else console.print("  [dim]Nothing to clean[/dim]")
            wait()

        elif choice == "Clean all":
            # Docs
            build_dir = REPO_ROOT / "docs" / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
            # Data
            data_dir = REPO_ROOT / "data"
            if data_dir.exists():
                for item in data_dir.iterdir():
                    if item.name not in ("README.md", ".gitkeep"):
                        shutil.rmtree(item) if item.is_dir() else item.unlink()
            # Caches
            for t in [".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"]:
                path = REPO_ROOT / t
                if path.exists():
                    shutil.rmtree(path)
            for pycache in REPO_ROOT.rglob("__pycache__"):
                shutil.rmtree(pycache)
            console.print("  [green]✓[/green] Cleaned all")
            wait()

        else:
            break


def menu_hpc():
    """HPC submenu."""
    from .hpc import discover_experiments, get_experiment_name

    while True:
        choice = select(
            "HPC:",
            [
                "Live monitor",
                "Job status",
                "Preview jobs (dry run)",
                "Submit jobs",
                "Kill job",
                "Kill all jobs",
                "← Back",
            ],
        )

        if choice == "Live monitor":
            hpc_monitor()

        elif choice == "Job status":
            hpc_status()
            wait()

        elif choice in ("Submit jobs", "Preview jobs (dry run)"):
            experiments = discover_experiments()
            if not experiments:
                console.print("  [dim]No experiments with jobs.yaml found[/dim]")
                wait()
                continue

            exp_names = ["all"] + [get_experiment_name(e) for e in experiments]
            experiment = select("Experiment:", exp_names + ["← Back"])

            if experiment and experiment != "← Back":
                dry_run = choice == "Preview jobs (dry run)"
                if not dry_run:
                    if not confirm("Submit to HPC?", default=False):
                        continue
                hpc_submit(experiment, dry_run)
                wait()

        elif choice == "Kill job":
            target = questionary.text(
                "Job name or ID:",
                style=STYLE,
            ).ask()
            if target:
                if confirm(f"Kill job '{target}'?", default=False):
                    hpc_kill(target)
                    wait()

        elif choice == "Kill all jobs":
            if confirm("Kill ALL your jobs?", default=False):
                hpc_kill("all")
                wait()

        else:
            break


def menu_code():
    """Code quality submenu."""
    while True:
        choice = select(
            "Code:",
            [
                "Lint (ruff check)",
                "Format (ruff format)",
                "← Back",
            ],
        )

        if choice == "Lint (ruff check)":
            ruff_check()
            wait()
        elif choice == "Format (ruff format)":
            ruff_format()
            wait()
        else:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────


def interactive():
    """Run interactive menu."""
    while True:
        console.clear()
        console.print(
            Panel.fit(
                "[bold cyan]ANA-P3[/bold cyan] [dim]Advanced Numerical Algorithms[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        choice = select(
            "Select:",
            [
                "Fetch MLflow data",
                "Runner",
                "Build docs",
                "Code",
                "Clean",
                "HPC",
                "Exit",
            ],
        )

        if choice is None or choice == "Exit":
            console.print("[dim]Goodbye![/dim]\n")
            break

        elif choice == "Fetch MLflow data":
            fetch_mlflow()
            wait()

        elif choice == "Runner":
            console.print()
            menu_runner()

        elif choice == "Build docs":
            build_docs()
            # Try to open in browser
            index = REPO_ROOT / "docs" / "build" / "html" / "index.html"
            if index.exists():
                import webbrowser

                webbrowser.open(f"file://{index}")
            wait()

        elif choice == "Code":
            console.print()
            menu_code()

        elif choice == "Clean":
            console.print()
            menu_clean()

        elif choice == "HPC":
            console.print()
            menu_hpc()
