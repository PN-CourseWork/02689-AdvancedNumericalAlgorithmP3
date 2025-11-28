"""Interactive CLI menu with arrow-key navigation."""

import questionary
from rich.panel import Panel

from .console import console
from .actions import (
    fetch_mlflow,
    run_scripts,
    build_docs,
    clean_all,
    ruff_check,
    ruff_format,
    hpc_submit,
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


def menu_data():
    """Data submenu."""
    while True:
        choice = select("Data:", [
            "Fetch MLflow artifacts",
            "Run compute scripts",
            "Run plot scripts",
            "← Back",
        ])

        if choice == "Fetch MLflow artifacts":
            fetch_mlflow()
            wait()
        elif choice == "Run compute scripts":
            run_scripts("compute")
            wait()
        elif choice == "Run plot scripts":
            run_scripts("plot")
            wait()
        else:
            break


def menu_code():
    """Code quality submenu."""
    while True:
        choice = select("Code:", [
            "Lint (ruff check)",
            "Format (ruff format)",
            "Lint + Format",
            "← Back",
        ])

        if choice == "Lint (ruff check)":
            ruff_check()
            wait()
        elif choice == "Format (ruff format)":
            ruff_format()
            wait()
        elif choice == "Lint + Format":
            ruff_check()
            ruff_format()
            wait()
        else:
            break


def menu_docs():
    """Documentation submenu."""
    while True:
        choice = select("Docs:", [
            "Build documentation",
            "Clean documentation",
            "← Back",
        ])

        if choice == "Build documentation":
            build_docs()
            wait()
        elif choice == "Clean documentation":
            import shutil
            from .actions import REPO_ROOT
            build_dir = REPO_ROOT / "docs" / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
                console.print("  [green]✓[/green] Cleaned docs/build")
            else:
                console.print("  [dim]Nothing to clean[/dim]")
            wait()
        else:
            break


def menu_hpc():
    """HPC submenu."""
    while True:
        choice = select("HPC:", [
            "Submit jobs",
            "Preview jobs (dry run)",
            "← Back",
        ])

        if choice in ("Submit jobs", "Preview jobs (dry run)"):
            solver = select("Solver:", ["all", "spectral", "fv", "← Back"])
            if solver and solver != "← Back":
                dry_run = choice == "Preview jobs (dry run)"
                if not dry_run:
                    if not confirm("Submit to HPC?", default=False):
                        continue
                hpc_submit(solver, dry_run)
                wait()
        else:
            break


def menu_clean():
    """Clean submenu."""
    while True:
        choice = select("Clean:", [
            "Clean all caches",
            "Clean documentation only",
            "Clean data only",
            "← Back",
        ])

        if choice == "Clean all caches":
            clean_all()
            wait()
        elif choice == "Clean documentation only":
            import shutil
            from .actions import REPO_ROOT
            build_dir = REPO_ROOT / "docs" / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
                console.print("  [green]✓[/green] Cleaned docs/build")
            else:
                console.print("  [dim]Nothing to clean[/dim]")
            wait()
        elif choice == "Clean data only":
            import shutil
            from .actions import REPO_ROOT
            data_dir = REPO_ROOT / "data"
            count = 0
            if data_dir.exists():
                for item in data_dir.iterdir():
                    if item.name not in ("README.md", ".gitkeep"):
                        shutil.rmtree(item) if item.is_dir() else item.unlink()
                        count += 1
            if count:
                console.print(f"  [green]✓[/green] Cleaned {count} items from data/")
            else:
                console.print("  [dim]Nothing to clean[/dim]")
            wait()
        else:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────


def interactive():
    """Run interactive menu with categories."""
    menus = {
        "Data": menu_data,
        "Code": menu_code,
        "Docs": menu_docs,
        "HPC": menu_hpc,
        "Clean": menu_clean,
    }

    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]ANA-P3[/bold cyan] [dim]Advanced Numerical Algorithms[/dim]",
            border_style="cyan",
        ))
        console.print()

        choice = select("Select category:", list(menus.keys()) + ["Exit"])

        if choice is None or choice == "Exit":
            console.print("[dim]Goodbye![/dim]\n")
            break

        if choice in menus:
            console.print()
            menus[choice]()
