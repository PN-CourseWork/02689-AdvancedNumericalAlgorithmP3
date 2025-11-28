"""Live HPC job monitor TUI."""

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
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

    for line in lines[1:]:  # Skip header
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


def kill_job(job_id: str) -> tuple[bool, str]:
    """Kill a job by ID."""
    try:
        result = subprocess.run(["bkill", job_id], capture_output=True, text=True)
        msg = result.stdout.strip() or result.stderr.strip()
        return result.returncode == 0, msg
    except Exception as e:
        return False, str(e)


def kill_all_jobs() -> tuple[bool, str]:
    """Kill all jobs."""
    try:
        result = subprocess.run(["bkill", "0"], capture_output=True, text=True)
        msg = result.stdout.strip() or result.stderr.strip()
        return result.returncode == 0, msg
    except Exception as e:
        return False, str(e)


def make_table(jobs: list[Job], selected: int = 0) -> Table:
    """Create jobs table."""
    table = Table(show_header=True, header_style="bold cyan", border_style="dim")

    table.add_column("#", style="dim", width=3)
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Queue")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Elapsed", justify="right")

    for i, job in enumerate(jobs):
        status_style = "green" if job.status == "RUN" else "yellow" if job.status == "PEND" else "dim"
        marker = ">" if i == selected else " "

        table.add_row(
            f"{marker}{i+1}",
            job.id,
            job.name,
            job.queue,
            Text(job.status, style=status_style),
            job.start_time,
            job.elapsed,
        )

    return table


def make_display(jobs: list[Job], selected: int, message: str) -> Group:
    """Create the full display."""
    if jobs:
        table = make_table(jobs, selected)
    else:
        table = Text("No running jobs", style="dim")

    panel = Panel(
        table,
        title="[bold cyan]HPC Monitor[/bold cyan]",
        subtitle=f"[dim]{len(jobs)} jobs | Refresh: 5s[/dim]",
        border_style="cyan",
    )

    help_text = Text.from_markup(
        "[dim]Commands:[/dim] "
        "[bold]1-9[/bold]=select  "
        "[bold]k[/bold]=kill selected  "
        "[bold]K[/bold]=kill all  "
        "[bold]r[/bold]=refresh  "
        "[bold]q[/bold]=quit"
    )

    parts = [panel, Text(""), help_text]

    if message:
        parts.append(Text(""))
        parts.append(Text(message, style="yellow"))

    parts.append(Text(""))
    parts.append(Text("> ", style="dim"))

    return Group(*parts)


def input_thread(cmd_queue: Queue, stop_event: threading.Event):
    """Thread to read input without blocking."""
    while not stop_event.is_set():
        try:
            # Use select for timeout on Unix
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                line = sys.stdin.readline()
                if line:
                    cmd_queue.put(line.strip().lower())
        except Exception:
            break


def monitor():
    """Run the HPC monitor."""
    console = Console()
    selected = 0
    message = ""
    running = True

    # Command queue for thread communication
    cmd_queue: Queue = Queue()
    stop_event = threading.Event()

    # Start input thread
    input_handler = threading.Thread(target=input_thread, args=(cmd_queue, stop_event), daemon=True)
    input_handler.start()

    # Initial job fetch
    jobs = get_jobs()

    try:
        with Live(console=console, refresh_per_second=4, screen=True) as live:
            last_refresh = time.time()

            while running:
                now = time.time()

                # Refresh jobs every 5 seconds
                if now - last_refresh >= 5:
                    jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))
                    last_refresh = now

                # Update display
                live.update(make_display(jobs, selected, message))
                message = ""  # Clear message after showing

                # Check for commands
                try:
                    cmd = cmd_queue.get_nowait()

                    if cmd == 'q':
                        running = False

                    elif cmd == 'r':
                        jobs = get_jobs()
                        selected = min(selected, max(0, len(jobs) - 1))
                        last_refresh = now
                        message = "Refreshed"

                    elif cmd == 'k' and jobs:
                        job = jobs[selected]
                        ok, msg = kill_job(job.id)
                        message = f"{'Killed' if ok else 'Failed'}: {job.name}"
                        jobs = get_jobs()
                        selected = min(selected, max(0, len(jobs) - 1))
                        last_refresh = now

                    elif cmd in ('K', 'ka'):
                        ok, msg = kill_all_jobs()
                        message = f"{'Killed all' if ok else 'Failed'}: {msg}"
                        jobs = get_jobs()
                        selected = min(selected, max(0, len(jobs) - 1))
                        last_refresh = now

                    elif cmd.isdigit():
                        num = int(cmd) - 1
                        if 0 <= num < len(jobs):
                            selected = num
                            message = f"Selected: {jobs[selected].name}"

                    elif cmd.startswith('k ') and len(cmd) > 2:
                        try:
                            num = int(cmd[2:]) - 1
                            if 0 <= num < len(jobs):
                                job = jobs[num]
                                ok, msg = kill_job(job.id)
                                message = f"{'Killed' if ok else 'Failed'}: {job.name}"
                                jobs = get_jobs()
                                selected = min(selected, max(0, len(jobs) - 1))
                                last_refresh = now
                        except ValueError:
                            message = "Invalid job number"

                except Empty:
                    pass

                time.sleep(0.1)  # Small delay to prevent CPU spinning

    finally:
        stop_event.set()
        console.clear()
        console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    monitor()
