"""Live HPC job monitor TUI."""

import subprocess
import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


@dataclass
class Job:
    """HPC job info."""
    id: str
    name: str
    queue: str
    status: str
    start_time: str
    elapsed: str


def parse_bstat_output(output: str) -> list[Job]:
    """Parse bstat output into Job objects."""
    jobs = []
    lines = output.strip().split("\n")

    # Skip header line
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 6:
            jobs.append(Job(
                id=parts[0],
                name=parts[3],
                queue=parts[2],
                status=parts[5],
                start_time=" ".join(parts[6:8]) if len(parts) > 7 else "",
                elapsed=parts[-1] if len(parts) > 8 else "",
            ))
    return jobs


def get_jobs() -> list[Job]:
    """Fetch current jobs from bstat."""
    try:
        result = subprocess.run(["bstat"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return parse_bstat_output(result.stdout)
    except Exception:
        pass
    return []


def kill_job(job_id: str) -> bool:
    """Kill a job by ID."""
    try:
        result = subprocess.run(["bkill", job_id], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def kill_all_jobs() -> bool:
    """Kill all jobs."""
    try:
        result = subprocess.run(["bkill", "0"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def make_table(jobs: list[Job], selected: int = 0) -> Table:
    """Create jobs table with selection highlight."""
    table = Table(
        title="HPC Jobs",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )

    table.add_column("", width=2)  # Selection indicator
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Queue")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Elapsed", justify="right")

    for i, job in enumerate(jobs):
        # Selection indicator
        indicator = "›" if i == selected else ""

        # Status color
        status_style = "green" if job.status == "RUN" else "yellow" if job.status == "PEND" else "dim"

        # Row style
        row_style = "reverse" if i == selected else ""

        table.add_row(
            indicator,
            job.id,
            job.name,
            job.queue,
            Text(job.status, style=status_style),
            job.start_time,
            job.elapsed,
            style=row_style,
        )

    return table


def make_help_bar() -> Text:
    """Create help bar at bottom."""
    help_text = Text()
    help_text.append(" ↑↓ ", style="black on white")
    help_text.append(" Navigate  ")
    help_text.append(" k ", style="black on white")
    help_text.append(" Kill selected  ")
    help_text.append(" K ", style="black on white")
    help_text.append(" Kill all  ")
    help_text.append(" r ", style="black on white")
    help_text.append(" Refresh  ")
    help_text.append(" q ", style="black on white")
    help_text.append(" Quit ")
    return help_text


def make_display(jobs: list[Job], selected: int, message: str = "") -> Panel:
    """Create full display panel."""
    layout = Layout()

    if jobs:
        content = make_table(jobs, selected)
    else:
        content = Text("No running jobs", style="dim", justify="center")

    # Add message if present
    if message:
        msg_text = Text(f"\n{message}", style="yellow")
        layout.split_column(
            Layout(content, name="table"),
            Layout(msg_text, name="message", size=2),
            Layout(make_help_bar(), name="help", size=1),
        )
    else:
        layout.split_column(
            Layout(content, name="table"),
            Layout(make_help_bar(), name="help", size=1),
        )

    return Panel(
        layout,
        title="[bold cyan]HPC Monitor[/bold cyan]",
        subtitle="[dim]Auto-refresh: 5s[/dim]",
        border_style="cyan",
    )


def monitor():
    """Run the live HPC monitor."""
    console = Console()

    # Check if we can use keyboard input
    try:
        import select
        import termios
        import tty
        has_tty = True
    except ImportError:
        has_tty = False

    if not has_tty:
        console.print("[red]Interactive monitor requires a TTY[/red]")
        return

    # Setup terminal for raw input
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setraw(sys.stdin.fileno())

        jobs = get_jobs()
        selected = 0
        message = ""
        last_refresh = time.time()

        with Live(make_display(jobs, selected, message), console=console, refresh_per_second=4) as live:
            while True:
                # Check for keyboard input (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)

                    if key == 'q':
                        break

                    elif key == 'r':
                        jobs = get_jobs()
                        selected = min(selected, max(0, len(jobs) - 1))
                        message = "Refreshed"

                    elif key == 'k' and jobs:
                        job = jobs[selected]
                        if kill_job(job.id):
                            message = f"Killed {job.name}"
                            time.sleep(0.5)
                            jobs = get_jobs()
                            selected = min(selected, max(0, len(jobs) - 1))
                        else:
                            message = f"Failed to kill {job.name}"

                    elif key == 'K':
                        if kill_all_jobs():
                            message = "Killed all jobs"
                            time.sleep(0.5)
                            jobs = get_jobs()
                            selected = 0
                        else:
                            message = "Failed to kill jobs"

                    # Arrow keys (escape sequences)
                    elif key == '\x1b':
                        # Read rest of escape sequence
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key2 = sys.stdin.read(1)
                            if key2 == '[':
                                if select.select([sys.stdin], [], [], 0.1)[0]:
                                    key3 = sys.stdin.read(1)
                                    if key3 == 'A' and jobs:  # Up
                                        selected = max(0, selected - 1)
                                        message = ""
                                    elif key3 == 'B' and jobs:  # Down
                                        selected = min(len(jobs) - 1, selected + 1)
                                        message = ""

                    live.update(make_display(jobs, selected, message))

                # Auto-refresh every 5 seconds
                if time.time() - last_refresh > 5:
                    jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))
                    last_refresh = time.time()
                    if not message.startswith("Killed") and not message.startswith("Failed"):
                        message = ""
                    live.update(make_display(jobs, selected, message))

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        console.print("\n[dim]Monitor closed[/dim]")


if __name__ == "__main__":
    monitor()
