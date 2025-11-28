"""Live HPC job monitor TUI using Textual."""

import subprocess
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane


@dataclass
class Job:
    """HPC job info."""
    id: str
    name: str
    user: str
    queue: str
    status: str
    start_time: str
    exec_host: str
    memory: str
    cpu_time: str


def parse_bjobs_output(output: str) -> list[Job]:
    """Parse bjobs -o output into Job objects."""
    jobs = []
    lines = output.strip().split("\n")

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 9:
            jobs.append(Job(
                id=parts[0],
                user=parts[1],
                status=parts[2],
                queue=parts[3],
                exec_host=parts[4] if parts[4] != "-" else "",
                name=parts[5],
                start_time=" ".join(parts[6:8]) if len(parts) > 7 else "",
                memory=parts[8] if len(parts) > 8 else "-",
                cpu_time=parts[9] if len(parts) > 9 else "-",
            ))
    return jobs


def get_jobs() -> list[Job]:
    """Fetch current jobs from bjobs with detailed info."""
    try:
        # Get detailed job info with specific columns
        result = subprocess.run(
            ["bjobs", "-o", "jobid user stat queue exec_host job_name submit_time memlimit cpu_used", "-noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return parse_bjobs_output("HEADER\n" + result.stdout)
    except Exception:
        pass

    # Fallback to bstat
    try:
        result = subprocess.run(["bstat"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            jobs = []
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 6:
                    jobs.append(Job(
                        id=parts[0],
                        user=parts[1] if len(parts) > 1 else "",
                        status=parts[5] if len(parts) > 5 else "",
                        queue=parts[2] if len(parts) > 2 else "",
                        exec_host="",
                        name=parts[3] if len(parts) > 3 else "",
                        start_time=" ".join(parts[6:8]) if len(parts) > 7 else "",
                        memory="-",
                        cpu_time="-",
                    ))
            return jobs
    except Exception:
        pass
    return []


def get_queue_info() -> str:
    """Get cluster queue information."""
    output_lines = []

    # Queue summary
    try:
        result = subprocess.run(
            ["bqueues"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output_lines.append("[#88c0d0 bold]Queue Summary[/]\n")
            output_lines.append(result.stdout)
    except Exception:
        output_lines.append("[#bf616a]Failed to get queue info[/]\n")

    # Host info
    try:
        result = subprocess.run(
            ["bhosts", "-w"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output_lines.append("\n[#88c0d0 bold]Host Status[/]\n")
            output_lines.append(result.stdout)
    except Exception:
        pass

    # Cluster load
    try:
        result = subprocess.run(
            ["lsload", "-w"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output_lines.append("\n[#88c0d0 bold]Cluster Load[/]\n")
            output_lines.append(result.stdout)
    except Exception:
        pass

    return "".join(output_lines) if output_lines else "No cluster information available"


def get_job_output(job_id: str, lines: int = 50) -> str:
    """Get the tail of a job's output file."""
    try:
        # Get output file path
        result = subprocess.run(
            ["bjobs", "-o", "output_file", "-noheader", job_id],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            output_file = result.stdout.strip()
            if output_file and output_file != "-":
                # Tail the output file
                tail_result = subprocess.run(
                    ["tail", "-n", str(lines), output_file],
                    capture_output=True, text=True, timeout=10
                )
                if tail_result.returncode == 0:
                    return f"[#88c0d0 bold]{output_file}[/]\n\n{tail_result.stdout}"
                return f"[#bf616a]Cannot read: {output_file}[/]"
        return "[#ebcb8b]No output file found[/]"
    except Exception as e:
        return f"[#bf616a]Error: {e}[/]"


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


class OutputView(Static):
    """View for job output."""

    def set_content(self, content: str) -> None:
        self.update(content)


class ResourceView(Static):
    """View for cluster resources."""

    def refresh_info(self) -> None:
        self.update(get_queue_info())


class HPCMonitorApp(App):
    """HPC Job Monitor TUI."""

    ENABLE_COMMAND_PALETTE = False

    # Nord theme colors
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

    TabbedContent {
        background: #2e3440;
    }

    TabPane {
        background: #2e3440;
        padding: 0;
    }

    Tabs {
        background: #3b4252;
    }

    Tab {
        background: #3b4252;
        color: #d8dee9;
    }

    Tab.-active {
        background: #434c5e;
        color: #88c0d0;
    }

    Tab:hover {
        background: #434c5e;
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

    #output-view {
        height: 1fr;
        border: round #88c0d0;
        background: #2e3440;
        color: #d8dee9;
        padding: 1;
    }

    #resource-view {
        height: 1fr;
        border: round #88c0d0;
        background: #2e3440;
        color: #d8dee9;
        padding: 1;
    }

    VerticalScroll {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("d", "kill_selected", "Kill"),
        Binding("D", "kill_all", "Kill All"),
        Binding("t", "tail_output", "Tail"),
        Binding("1", "show_jobs", "Jobs", show=False),
        Binding("2", "show_output", "Output", show=False),
        Binding("3", "show_resources", "Resources", show=False),
        Binding("up,k", "cursor_up", "Up", show=False),
        Binding("down,j", "cursor_down", "Down", show=False),
    ]

    TITLE = "HPC Monitor"

    def __init__(self):
        super().__init__()
        self.jobs: list[Job] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Jobs", id="jobs-tab"):
                yield DataTable(id="job-table")
            with TabPane("Output", id="output-tab"):
                with VerticalScroll():
                    yield OutputView(id="output-view")
            with TabPane("Resources", id="resources-tab"):
                with VerticalScroll():
                    yield ResourceView(id="resource-view")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the table and start auto-refresh."""
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Name", "Queue", "Status", "Host", "Memory", "CPU", "Started")

        self.refresh_jobs()
        self.set_interval(5, self.refresh_jobs)

    def refresh_jobs(self) -> None:
        """Refresh the job list."""
        self.jobs = get_jobs()
        table = self.query_one(DataTable)

        cursor_row = table.cursor_row if table.row_count > 0 else 0
        table.clear()

        if self.jobs:
            for job in self.jobs:
                # Nord Aurora colors
                status_style = (
                    "#a3be8c" if job.status == "RUN"
                    else "#ebcb8b" if job.status == "PEND"
                    else "#4c566a"
                )
                # Truncate name if too long
                name = job.name[:30] + "..." if len(job.name) > 33 else job.name
                table.add_row(
                    job.id,
                    name,
                    job.queue,
                    f"[{status_style}]{job.status}[/{status_style}]",
                    job.exec_host or "-",
                    job.memory,
                    job.cpu_time,
                    job.start_time,
                )
            if cursor_row < table.row_count:
                table.move_cursor(row=cursor_row)

    def action_refresh(self) -> None:
        """Manual refresh."""
        self.refresh_jobs()
        # Also refresh resources if on that tab
        self.query_one(ResourceView).refresh_info()
        self.query_one(StatusBar).set_message("Refreshed")

    def action_tail_output(self) -> None:
        """Show output of selected job."""
        if not self.jobs:
            self.query_one(StatusBar).set_message("No jobs", "#bf616a")
            return

        table = self.query_one(DataTable)
        row = table.cursor_row

        if 0 <= row < len(self.jobs):
            job = self.jobs[row]
            output = get_job_output(job.id)
            self.query_one(OutputView).set_content(output)
            # Switch to output tab
            self.query_one(TabbedContent).active = "output-tab"
            self.query_one(StatusBar).set_message(f"Output: {job.name}")

    def action_show_jobs(self) -> None:
        """Switch to jobs tab."""
        self.query_one(TabbedContent).active = "jobs-tab"

    def action_show_output(self) -> None:
        """Switch to output tab."""
        self.query_one(TabbedContent).active = "output-tab"

    def action_show_resources(self) -> None:
        """Switch to resources tab."""
        self.query_one(ResourceView).refresh_info()
        self.query_one(TabbedContent).active = "resources-tab"

    def action_kill_selected(self) -> None:
        """Kill the selected job."""
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
