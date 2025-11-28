"""Live HPC job monitor TUI using Textual."""

import subprocess
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import DataTable, Footer, Header, Static


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


class StatusBar(Static):
    """Status bar for messages."""

    # Nord Aurora yellow: #ebcb8b
    def set_message(self, msg: str, style: str = "#ebcb8b") -> None:
        self.update(f"[{style}]{msg}[/{style}]" if msg else "")


class HPCMonitorApp(App):
    """HPC Job Monitor TUI."""

    ENABLE_COMMAND_PALETTE = False

    # Nord theme colors
    # Polar Night: #2e3440, #3b4252, #434c5e, #4c566a
    # Snow Storm: #d8dee9, #e5e9f0, #eceff4
    # Frost: #8fbcbb, #88c0d0, #81a1c1, #5e81ac
    # Aurora: #bf616a (red), #ebcb8b (yellow), #a3be8c (green)

    CSS = """
    Screen {
        background: #2e3440;
    }

    Header {
        background: #3b4252;
        color: #88c0d0;
    }

    Footer {
        background: #3b4252;
        color: #d8dee9;
    }

    Footer > .footer--key {
        background: #5e81ac;
        color: #eceff4;
    }

    Footer > .footer--description {
        color: #d8dee9;
    }

    #job-table {
        height: 1fr;
        border: round #88c0d0;
        background: #2e3440;
    }

    DataTable {
        background: #2e3440;
    }

    DataTable > .datatable--header {
        background: #3b4252;
        color: #88c0d0;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #434c5e;
        color: #eceff4;
    }

    DataTable > .datatable--hover {
        background: #3b4252;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: #2e3440;
        color: #d8dee9;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("d", "kill_selected", "Kill Job"),
        Binding("D", "kill_all", "Kill All"),
        Binding("up,k", "cursor_up", "Up", show=False),
        Binding("down,j", "cursor_down", "Down", show=False),
    ]

    TITLE = "HPC Monitor"

    def __init__(self):
        super().__init__()
        self.jobs: list[Job] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            DataTable(id="job-table"),
        )
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the table and start auto-refresh."""
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "ID", "Name", "Queue", "Status", "Started", "Elapsed")

        self.refresh_jobs()
        self.set_interval(5, self.refresh_jobs)

    def refresh_jobs(self) -> None:
        """Refresh the job list."""
        self.jobs = get_jobs()
        table = self.query_one(DataTable)

        # Remember cursor position
        cursor_row = table.cursor_row if table.row_count > 0 else 0

        table.clear()

        if self.jobs:
            for i, job in enumerate(self.jobs):
                # Nord Aurora: green=#a3be8c, yellow=#ebcb8b, dim=#4c566a
                status_style = (
                    "#a3be8c" if job.status == "RUN"
                    else "#ebcb8b" if job.status == "PEND"
                    else "#4c566a"
                )
                table.add_row(
                    str(i + 1),
                    job.id,
                    job.name,
                    job.queue,
                    f"[{status_style}]{job.status}[/{status_style}]",
                    job.start_time,
                    job.elapsed,
                )
            # Restore cursor position
            if cursor_row < table.row_count:
                table.move_cursor(row=cursor_row)

    def action_refresh(self) -> None:
        """Manual refresh."""
        self.refresh_jobs()
        self.query_one(StatusBar).set_message("Refreshed")

    def action_kill_selected(self) -> None:
        """Kill the selected job."""
        # Nord Aurora: red=#bf616a, green=#a3be8c
        if not self.jobs:
            self.query_one(StatusBar).set_message("No jobs to kill", "#bf616a")
            return

        table = self.query_one(DataTable)
        row = table.cursor_row

        if 0 <= row < len(self.jobs):
            job = self.jobs[row]
            ok, msg = kill_job(job.id)
            status = self.query_one(StatusBar)
            if ok:
                status.set_message(f"Killed: {job.name}", "#a3be8c")
            else:
                status.set_message(f"Failed to kill {job.name}: {msg}", "#bf616a")
            self.refresh_jobs()

    def action_kill_all(self) -> None:
        """Kill all jobs."""
        # Nord Aurora: red=#bf616a, green=#a3be8c
        ok, msg = kill_all_jobs()
        status = self.query_one(StatusBar)
        if ok:
            status.set_message("Killed all jobs", "#a3be8c")
        else:
            status.set_message(f"Failed: {msg}", "#bf616a")
        self.refresh_jobs()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        table = self.query_one(DataTable)
        table.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        table = self.query_one(DataTable)
        table.action_cursor_down()


def monitor():
    """Run the HPC monitor."""
    app = HPCMonitorApp()
    app.run()


if __name__ == "__main__":
    monitor()
