"""Live HPC job monitor TUI using blessed."""

import subprocess
from dataclasses import dataclass

from blessed import Terminal


@dataclass
class Job:
    """HPC job info."""
    id: str
    name: str
    queue: str
    status: str
    exec_host: str


def get_jobs() -> list[Job]:
    """Fetch current jobs from bjobs."""
    try:
        # Simple bjobs call - most compatible
        result = subprocess.run(
            ["bjobs", "-w"],  # Wide format
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            jobs = []
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 7:
                    # Format: JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME ...
                    jobs.append(Job(
                        id=parts[0],
                        status=parts[2],
                        queue=parts[3],
                        exec_host=parts[5] if parts[5] != "-" else "",
                        name=parts[6],
                    ))
            return jobs
    except Exception:
        pass

    # Fallback to bstat
    try:
        result = subprocess.run(["bstat"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            jobs = []
            for line in result.stdout.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) >= 6:
                    jobs.append(Job(
                        id=parts[0],
                        status=parts[5] if len(parts) > 5 else "",
                        queue=parts[2] if len(parts) > 2 else "",
                        exec_host="",
                        name=parts[3] if len(parts) > 3 else "",
                    ))
            return jobs
    except Exception:
        pass
    return []


def get_queue_info() -> list[str]:
    """Get cluster queue information."""
    lines = []
    try:
        result = subprocess.run(["bqueues"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines.append("=== Queue Summary ===")
            lines.extend(result.stdout.strip().split("\n"))
    except Exception:
        lines.append("Failed to get queue info")

    try:
        result = subprocess.run(["bhosts", "-w"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines.append("")
            lines.append("=== Host Status ===")
            lines.extend(result.stdout.strip().split("\n"))
    except Exception:
        pass

    return lines


def get_job_details(job_id: str) -> list[str]:
    """Get detailed job information using bjobs -l."""
    lines = []
    try:
        # Get full job details
        result = subprocess.run(
            ["bjobs", "-l", job_id],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append("=== Job Details ===")
            lines.append("")
            # bjobs -l output has weird formatting, clean it up
            for line in result.stdout.split("\n"):
                # Remove excessive whitespace but keep structure
                cleaned = " ".join(line.split())
                if cleaned:
                    lines.append(cleaned)
            lines.append("")
    except Exception as e:
        lines.append(f"Error getting job details: {e}")

    # Try to get output file content
    try:
        # First find the output file
        peek = subprocess.run(
            ["bjobs", "-o", "output_file", "-noheader", job_id],
            capture_output=True, text=True, timeout=10
        )
        if peek.returncode == 0:
            output_file = peek.stdout.strip()
            if output_file and output_file not in ("-", ""):
                lines.append("=== Output File ===")
                lines.append(f"File: {output_file}")
                lines.append("")
                # Try to tail the file
                tail = subprocess.run(
                    ["tail", "-n", "50", output_file],
                    capture_output=True, text=True, timeout=10
                )
                if tail.returncode == 0:
                    lines.extend(tail.stdout.split("\n"))
                else:
                    lines.append("(file not yet available or not readable)")
    except Exception:
        pass

    return lines if lines else ["No information available"]


def kill_job(job_id: str) -> tuple[bool, str]:
    """Kill a job."""
    try:
        result = subprocess.run(["bkill", job_id], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return False, str(e)


def kill_all_jobs() -> tuple[bool, str]:
    """Kill all jobs."""
    try:
        result = subprocess.run(["bkill", "0"], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return False, str(e)


def draw_floating_window(term, title: str, lines: list[str], scroll: int) -> None:
    """Draw a centered floating window with content."""
    # Window dimensions
    win_width = min(term.width - 4, 100)
    win_height = min(term.height - 6, 30)
    start_x = (term.width - win_width) // 2
    start_y = (term.height - win_height) // 2

    # Top border
    print(term.move_xy(start_x, start_y) + term.cyan + "╭" + "─" * (win_width - 2) + "╮" + term.normal)

    # Title bar
    title_text = f" {title} "
    padding = win_width - 4 - len(title_text)
    print(term.move_xy(start_x, start_y + 1) + term.cyan + "│" + term.normal +
          term.bold + title_text + term.normal + " " * padding +
          term.bright_black + "[j/k scroll, Esc close]" + term.normal +
          term.cyan + " │" + term.normal)
    print(term.move_xy(start_x, start_y + 2) + term.cyan + "├" + "─" * (win_width - 2) + "┤" + term.normal)

    # Content
    content_height = win_height - 4
    visible = lines[scroll:scroll + content_height]

    for i in range(content_height):
        line_content = visible[i][:win_width - 4] if i < len(visible) else ""
        padding = win_width - 4 - len(line_content)
        print(term.move_xy(start_x, start_y + 3 + i) +
              term.cyan + "│ " + term.normal + line_content + " " * padding + term.cyan + " │" + term.normal)

    # Bottom border
    print(term.move_xy(start_x, start_y + win_height - 1) + term.cyan + "╰" + "─" * (win_width - 2) + "╯" + term.normal)


def monitor():
    """Run the HPC monitor."""
    term = Terminal()
    selected = 0
    message = ""
    views = ["all", "running", "pending", "resources"]
    view_idx = 0
    output_lines: list[str] = []
    resource_lines: list[str] = []
    scroll = 0
    modal_open = False
    modal_scroll = 0

    all_jobs = get_jobs()

    with term.fullscreen(), term.cbreak(), term.hidden_cursor():
        while True:
            view = views[view_idx]

            # Filter jobs based on view
            if view == "running":
                jobs = [j for j in all_jobs if j.status == "RUN"]
            elif view == "pending":
                jobs = [j for j in all_jobs if j.status == "PEND"]
            else:
                jobs = all_jobs

            # Draw screen
            print(term.home + term.clear)

            # Header with tabs
            tabs = []
            for i, v in enumerate(views):
                label = v.upper()
                if v == "all":
                    label = f"ALL ({len(all_jobs)})"
                elif v == "running":
                    label = f"RUN ({len([j for j in all_jobs if j.status == 'RUN'])})"
                elif v == "pending":
                    label = f"PEND ({len([j for j in all_jobs if j.status == 'PEND'])})"

                if i == view_idx:
                    tabs.append(term.reverse + f" {label} " + term.normal)
                else:
                    tabs.append(f" {label} ")

            tab_line = " │ ".join(tabs)
            print(term.bold + term.cyan + " HPC Monitor " + term.normal + "  " + tab_line)
            print(term.cyan + "─" * term.width + term.normal)

            if view in ("all", "running", "pending"):
                # Full width job view
                hdr = f"{'ID':<12} {'Name':<50} {'Queue':<10} {'Status':<8} {'Host':<20}"
                print(term.bold + hdr[:term.width] + term.normal)
                print(term.bright_black + "─" * term.width + term.normal)

                max_rows = term.height - 8
                for i, job in enumerate(jobs[:max_rows]):
                    name = job.name[:48] + ".." if len(job.name) > 50 else job.name
                    host = job.exec_host[:18] + ".." if len(job.exec_host) > 20 else (job.exec_host or "-")

                    if job.status == "RUN":
                        status = term.green + f"{job.status:<8}" + term.normal
                    elif job.status == "PEND":
                        status = term.yellow + f"{job.status:<8}" + term.normal
                    else:
                        status = term.bright_black + f"{job.status:<8}" + term.normal

                    row = f"{job.id:<12} {name:<50} {job.queue:<10} {status} {host:<20}"

                    if i == selected:
                        print(term.reverse + row[:term.width] + term.normal)
                    else:
                        print(row[:term.width])

                if not jobs:
                    print(term.bright_black + "  No jobs" + term.normal)

            elif view == "resources":
                max_rows = term.height - 6
                visible = resource_lines[scroll:scroll + max_rows]
                for line in visible:
                    print(line[:term.width])

            # Footer
            print(term.move_y(term.height - 3) + term.cyan + "─" * term.width + term.normal)

            if message:
                print(term.yellow + f" {message}" + term.normal)
            else:
                print()

            help_text = " [h/l] Tabs  [j/k] Navigate  [t] Job details  [d] Kill  [D] Kill All  [r] Refresh  [q] Quit"
            print(term.bright_black + help_text[:term.width] + term.normal)

            # Draw floating modal if open
            if modal_open and output_lines:
                job_name = jobs[selected].name if jobs and 0 <= selected < len(jobs) else "Output"
                draw_floating_window(term, job_name, output_lines, modal_scroll)

            # Input with timeout for auto-refresh
            key = term.inkey(timeout=5)

            if key:
                message = ""

                # Modal controls
                if modal_open:
                    if key.name == 'KEY_ESCAPE' or key == 't' or key == 'q':
                        modal_open = False
                    elif key == 'j' or key.name == 'KEY_DOWN':
                        modal_scroll = min(modal_scroll + 1, max(0, len(output_lines) - 10))
                    elif key == 'k' or key.name == 'KEY_UP':
                        modal_scroll = max(0, modal_scroll - 1)
                    elif key == 'G':  # Jump to bottom
                        modal_scroll = max(0, len(output_lines) - 10)
                    elif key == 'g':  # Jump to top
                        modal_scroll = 0
                    elif key.lower() == 'r':  # Refresh output
                        if jobs and 0 <= selected < len(jobs):
                            output_lines = get_job_details(jobs[selected].id)
                            message = "Refreshed"
                    continue

                if key.lower() == 'q':
                    break

                elif key.lower() == 'r':
                    all_jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))
                    if view == "resources":
                        resource_lines = get_queue_info()
                    message = "Refreshed"

                # Tab navigation
                elif key == 'h' or key.name == 'KEY_LEFT':
                    view_idx = (view_idx - 1) % len(views)
                    selected = 0
                    scroll = 0
                    if views[view_idx] == "resources":
                        resource_lines = get_queue_info()

                elif key == 'l' or key.name == 'KEY_RIGHT':
                    view_idx = (view_idx + 1) % len(views)
                    selected = 0
                    scroll = 0
                    if views[view_idx] == "resources":
                        resource_lines = get_queue_info()

                # Vertical navigation
                elif key == 'j' or key.name == 'KEY_DOWN':
                    if view in ("all", "running", "pending"):
                        selected = min(selected + 1, len(jobs) - 1) if jobs else 0
                    elif view == "resources":
                        scroll += 1

                elif key == 'k' or key.name == 'KEY_UP':
                    if view in ("all", "running", "pending"):
                        selected = max(selected - 1, 0)
                    elif view == "resources":
                        scroll = max(0, scroll - 1)

                elif key == 'd' and jobs and view in ("all", "running", "pending"):
                    job = jobs[selected]
                    ok, msg = kill_job(job.id)
                    message = f"Killed: {job.name}" if ok else f"Failed: {msg}"
                    all_jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))

                elif key == 'D':
                    ok, msg = kill_all_jobs()
                    message = "Killed all jobs" if ok else f"Failed: {msg}"
                    all_jobs = get_jobs()
                    selected = 0

                # Open job details modal
                elif key == 't' and jobs and view in ("all", "running", "pending"):
                    output_lines = get_job_details(jobs[selected].id)
                    modal_scroll = 0
                    modal_open = True

            else:
                # Auto-refresh on timeout (but not if modal is open)
                if not modal_open:
                    all_jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))


if __name__ == "__main__":
    monitor()
