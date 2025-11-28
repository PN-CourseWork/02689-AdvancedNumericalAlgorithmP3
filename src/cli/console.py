"""Rich console output helpers."""

from rich.console import Console

console = Console()


def ok(msg: str):
    """Print success message."""
    console.print(f"  [green]✓[/green] {msg}")


def fail(msg: str):
    """Print failure message."""
    console.print(f"  [red]✗[/red] {msg}")


def dim(msg: str):
    """Print dimmed message."""
    console.print(f"  [dim]{msg}[/dim]")


def header(msg: str):
    """Print bold header."""
    console.print(f"\n[bold]{msg}[/bold]")
