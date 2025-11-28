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
    memory: str
    cpu_time: str
    start_time: str


def get_jobs() -> list[Job]:
    """Fetch current jobs from bjobs."""
    try:
        result = subprocess.run(
            ["bjobs", "-o", "jobid stat queue exec_host job_name submit_time memlimit cpu_used", "-noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            jobs = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 6:
                    jobs.append(Job(
                        id=parts[0],
                        status=parts[1],
                        queue=parts[2],
                        exec_host=parts[3] if parts[3] != "-" else "",
                        name=parts[4],
                        start_time=" ".join(parts[5:7]) if len(parts) > 6 else "",
                        memory=parts[7] if len(parts) > 7 else "-",
                        cpu_time=parts[8] if len(parts) > 8 else "-",
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
                        start_time=" ".join(parts[6:8]) if len(parts) > 7 else "",
                        memory="-",
                        cpu_time="-",
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


def get_job_output(job_id: str, num_lines: int = 30) -> list[str]:
    """Get tail of job output."""
    try:
        result = subprocess.run(
            ["bjobs", "-o", "output_file", "-noheader", job_id],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            output_file = result.stdout.strip()
            if output_file and output_file != "-":
                tail = subprocess.run(
                    ["tail", "-n", str(num_lines), output_file],
                    capture_output=True, text=True, timeout=10
                )
                if tail.returncode == 0:
                    return [f"=== {output_file} ===", ""] + tail.stdout.split("\n")
                return [f"Cannot read: {output_file}"]
        return ["No output file found"]
    except Exception as e:
        return [f"Error: {e}"]


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


def monitor():
    """Run the HPC monitor."""
    term = Terminal()
    selected = 0
    message = ""
    views = ["all", "running", "pending", "resources"]
    view_idx = 0
    output_lines: list[str] = []
    resource_lines: list[str] = []
    output_scroll = 0
    show_output = False  # Toggle for split view

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
            output_indicator = "  " + term.green + "[OUTPUT]" + term.normal if show_output else ""
            print(term.bold + term.cyan + " HPC Monitor " + term.normal + "  " + tab_line + output_indicator)
            print(term.cyan + "─" * term.width + term.normal)

            if view in ("all", "running", "pending"):
                max_rows = term.height - 8

                if show_output:
                    # Split view: jobs on left, output on right
                    left_width = term.width // 2 - 1
                    right_width = term.width - left_width - 3

                    # Column headers (compact for split view)
                    hdr = f"{'ID':<8} {'Name':<20} {'St':<4}"
                    out_hdr = " Output"
                    print(term.bold + hdr[:left_width] + term.normal + " │ " + term.bold + out_hdr + term.normal)
                    print(term.bright_black + "─" * left_width + "─┼─" + "─" * right_width + term.normal)

                    # Prepare output lines for right pane
                    visible_output = output_lines[output_scroll:output_scroll + max_rows] if output_lines else ["(press 't' on a job)"]

                    for i in range(max_rows):
                        # Left side: job row
                        if i < len(jobs):
                            job = jobs[i]
                            name = job.name[:18] + ".." if len(job.name) > 20 else job.name

                            if job.status == "RUN":
                                status = term.green + f"{job.status:<4}" + term.normal
                            elif job.status == "PEND":
                                status = term.yellow + f"{job.status:<4}" + term.normal
                            else:
                                status = term.bright_black + f"{job.status:<4}" + term.normal

                            left = f"{job.id:<8} {name:<20} {status}"
                            if i == selected:
                                left = term.reverse + f"{job.id:<8} {name:<20} " + term.normal + status
                        else:
                            left = " " * left_width

                        # Right side: output line
                        if i < len(visible_output):
                            right = visible_output[i][:right_width]
                        else:
                            right = ""

                        print(f"{left:<{left_width}} │ {right}")

                else:
                    # Full width job view
                    hdr = f"{'ID':<10} {'Name':<35} {'Queue':<8} {'Status':<6} {'Host':<12} {'Mem':<8} {'CPU':<10} {'Started':<16}"
                    print(term.bold + hdr[:term.width] + term.normal)
                    print(term.bright_black + "─" * term.width + term.normal)

                    for i, job in enumerate(jobs[:max_rows]):
                        name = job.name[:33] + ".." if len(job.name) > 35 else job.name
                        host = job.exec_host[:10] + ".." if len(job.exec_host) > 12 else job.exec_host

                        if job.status == "RUN":
                            status = term.green + f"{job.status:<6}" + term.normal
                        elif job.status == "PEND":
                            status = term.yellow + f"{job.status:<6}" + term.normal
                        else:
                            status = term.bright_black + f"{job.status:<6}" + term.normal

                        row = f"{job.id:<10} {name:<35} {job.queue:<8} {status} {host:<12} {job.memory:<8} {job.cpu_time:<10} {job.start_time:<16}"

                        if i == selected:
                            print(term.reverse + row[:term.width] + term.normal)
                        else:
                            print(row[:term.width])

                    if not jobs:
                        print(term.bright_black + "  No jobs" + term.normal)

            elif view == "resources":
                max_rows = term.height - 6
                visible = resource_lines[output_scroll:output_scroll + max_rows]
                for line in visible:
                    print(line[:term.width])

            # Footer
            print(term.move_y(term.height - 3) + term.cyan + "─" * term.width + term.normal)

            if message:
                print(term.yellow + f" {message}" + term.normal)
            else:
                print()

            if show_output:
                help_text = " [h/l] Tabs  [j/k] Jobs  [J/K] Scroll output  [t] Close  [d] Kill  [r] Refresh  [q] Quit"
            else:
                help_text = " [h/l] Tabs  [j/k] Navigate  [t] Show output  [d] Kill  [D] Kill All  [r] Refresh  [q] Quit"
            print(term.bright_black + help_text[:term.width] + term.normal)

            # Input with timeout for auto-refresh
            key = term.inkey(timeout=5)

            if key:
                message = ""

                if key.lower() == 'q':
                    break

                elif key.lower() == 'r':
                    all_jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))
                    if view == "resources":
                        resource_lines = get_queue_info()
                    # Refresh output if showing
                    if show_output and jobs and 0 <= selected < len(jobs):
                        output_lines = get_job_output(jobs[selected].id, num_lines=100)
                    message = "Refreshed"

                # Tab navigation with h/l or left/right arrows
                elif key == 'h' or key.name == 'KEY_LEFT':
                    view_idx = (view_idx - 1) % len(views)
                    selected = 0
                    output_scroll = 0
                    if views[view_idx] == "resources":
                        resource_lines = get_queue_info()

                elif key == 'l' or key.name == 'KEY_RIGHT':
                    view_idx = (view_idx + 1) % len(views)
                    selected = 0
                    output_scroll = 0
                    if views[view_idx] == "resources":
                        resource_lines = get_queue_info()

                # Scroll output pane with J/K (uppercase) or Shift+arrows
                elif key == 'J' or key.name == 'KEY_SDOWN':
                    if show_output:
                        output_scroll = min(output_scroll + 1, max(0, len(output_lines) - 5))
                    elif view == "resources":
                        output_scroll += 1

                elif key == 'K' or key.name == 'KEY_SUP':
                    if show_output or view == "resources":
                        output_scroll = max(0, output_scroll - 1)

                # Vertical navigation with j/k or up/down arrows
                elif key == 'j' or key.name == 'KEY_DOWN':
                    if view in ("all", "running", "pending"):
                        old_selected = selected
                        selected = min(selected + 1, len(jobs) - 1) if jobs else 0
                        # Auto-update output when selection changes
                        if show_output and selected != old_selected and jobs:
                            output_lines = get_job_output(jobs[selected].id, num_lines=100)
                            output_scroll = 0
                    elif view == "resources":
                        output_scroll += 1

                elif key == 'k' or key.name == 'KEY_UP':
                    if view in ("all", "running", "pending"):
                        old_selected = selected
                        selected = max(selected - 1, 0)
                        # Auto-update output when selection changes
                        if show_output and selected != old_selected and jobs:
                            output_lines = get_job_output(jobs[selected].id, num_lines=100)
                            output_scroll = 0
                    elif view == "resources":
                        output_scroll = max(0, output_scroll - 1)

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

                # Toggle output pane
                elif key == 't' and view in ("all", "running", "pending"):
                    if show_output:
                        show_output = False
                        message = ""
                    elif jobs:
                        show_output = True
                        output_lines = get_job_output(jobs[selected].id, num_lines=100)
                        output_scroll = 0
                        message = f"Output: {jobs[selected].name}"

            else:
                # Auto-refresh on timeout
                all_jobs = get_jobs()
                selected = min(selected, max(0, len(jobs) - 1))


if __name__ == "__main__":
    monitor()
