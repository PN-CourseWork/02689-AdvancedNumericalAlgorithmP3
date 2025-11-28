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

MENU_STYLE = questionary.Style([
    ("highlighted", "bold cyan"),
    ("pointer", "cyan"),
])


def interactive():
    """Run interactive menu with arrow-key navigation."""
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
            style=MENU_STYLE,
        ).ask()

        if choice is None or choice == "Exit":
            console.print("[dim]Goodbye![/dim]\n")
            break

        for name, action in actions:
            if name == choice:
                if action == "hpc":
                    _hpc_submenu()
                elif action:
                    action()
                    input("\nEnter to continue...")
                break


def _hpc_submenu():
    """HPC job submission submenu."""
    solver = questionary.select(
        "Solver:",
        choices=["all", "spectral", "fv", "← Back"],
    ).ask()

    if solver and solver != "← Back":
        dry_run = questionary.confirm("Dry run?", default=True).ask()
        if dry_run is not None:
            hpc_submit(solver, dry_run)
            input("\nEnter to continue...")
