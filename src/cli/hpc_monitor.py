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
    view = "jobs"  # jobs, output, resources
    output_lines: list[str] = []
    resource_lines: list[str] = []
    scroll_offset = 0

    jobs = get_jobs()

    with term.fullscreen(), term.cbreak(), term.hidden_cursor():
        while True:
            # Draw screen
            print(term.home + term.clear)

            # Header
            header = f" HPC Monitor | View: {view.upper()} | {len(jobs)} jobs "
            print(term.center(term.bold + term.cyan + header + term.normal))
            print(term.cyan + "─" * term.width + term.normal)

            if view == "jobs":
                # Column headers
                hdr = f"{'ID':<10} {'Name':<35} {'Queue':<8} {'Status':<6} {'Host':<12} {'Mem':<8} {'CPU':<10} {'Started':<16}"
                print(term.bold + hdr[:term.width] + term.normal)
                print(term.bright_black + "─" * term.width + term.normal)

                # Job rows
                max_rows = term.height - 8
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
                    print(term.bright_black + "  No running jobs" + term.normal)

            elif view == "output":
                max_rows = term.height - 6
                visible = output_lines[scroll_offset:scroll_offset + max_rows]
                for line in visible:
                    print(line[:term.width])

            elif view == "resources":
                max_rows = term.height - 6
                visible = resource_lines[scroll_offset:scroll_offset + max_rows]
                for line in visible:
                    print(line[:term.width])

            # Footer
            print(term.move_y(term.height - 3) + term.cyan + "─" * term.width + term.normal)

            if message:
                print(term.yellow + f" {message}" + term.normal)
            else:
                print()

            help_text = " [j/k] Navigate  [d] Kill  [D] Kill All  [t] Tail  [o] Output  [R] Resources  [r] Refresh  [q] Quit"
            print(term.bright_black + help_text[:term.width] + term.normal)

            # Input with timeout for auto-refresh
            key = term.inkey(timeout=5)

            if key:
                message = ""

                if key.lower() == 'q':
                    break

                elif key.lower() == 'r':
                    jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))
                    if view == "resources":
                        resource_lines = get_queue_info()
                    message = "Refreshed"

                elif key == 'j' or key.name == 'KEY_DOWN':
                    if view == "jobs":
                        selected = min(selected + 1, len(jobs) - 1) if jobs else 0
                    else:
                        scroll_offset += 1

                elif key == 'k' or key.name == 'KEY_UP':
                    if view == "jobs":
                        selected = max(selected - 1, 0)
                    else:
                        scroll_offset = max(0, scroll_offset - 1)

                elif key == 'd' and jobs and view == "jobs":
                    job = jobs[selected]
                    ok, msg = kill_job(job.id)
                    message = f"Killed: {job.name}" if ok else f"Failed: {msg}"
                    jobs = get_jobs()
                    selected = min(selected, max(0, len(jobs) - 1))

                elif key == 'D':
                    ok, msg = kill_all_jobs()
                    message = "Killed all jobs" if ok else f"Failed: {msg}"
                    jobs = get_jobs()
                    selected = 0

                elif key == 't' and jobs:
                    job = jobs[selected]
                    output_lines = get_job_output(job.id)
                    scroll_offset = 0
                    view = "output"
                    message = f"Output: {job.name}"

                elif key == 'o':
                    view = "output"
                    scroll_offset = 0

                elif key == 'R':
                    resource_lines = get_queue_info()
                    scroll_offset = 0
                    view = "resources"

                elif key == '1':
                    view = "jobs"
                    scroll_offset = 0

                elif key == '2':
                    view = "output"
                    scroll_offset = 0

                elif key == '3':
                    resource_lines = get_queue_info()
                    view = "resources"
                    scroll_offset = 0

            else:
                # Auto-refresh on timeout
                jobs = get_jobs()
                selected = min(selected, max(0, len(jobs) - 1))


if __name__ == "__main__":
    monitor()
