#!/usr/bin/env python3
"""Main script to run examples and submit HPC job packs."""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

def get_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current

REPO_ROOT = get_repo_root()

def discover_scripts():
    experiments_dir = REPO_ROOT / "Experiments"
    if not experiments_dir.exists():
        return [], []
    scripts = [
        p for p in experiments_dir.rglob("*.py")
        if p.is_file() and p.name != "__init__.py"
    ]
    compute_scripts = sorted([s for s in scripts if "compute" in s.name])
    plot_scripts = sorted([s for s in scripts if "plot" in s.name])
    return compute_scripts, plot_scripts

def run_scripts(scripts):
    if not scripts:
        print("  No scripts to run")
        return
    print(f"\nRunning {len(scripts)} scripts...\n")
    success_count = fail_count = 0
    for script in scripts:
        display_path = script.relative_to(REPO_ROOT)
        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(REPO_ROOT),
            )
            if result.returncode == 0:
                print(f"  ✓ {display_path}")
                success_count += 1
            else:
                print(f"  ✗ {display_path} (exit {result.returncode})")
                if result.stderr:
                    print(f"    Error: {result.stderr[:300]}")
                fail_count += 1
        except subprocess.TimeoutExpired:
            print(f"  ✗ {display_path} (timeout)")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ {display_path} ({e})")
            fail_count += 1
    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")

def build_docs():
    docs_dir = REPO_ROOT / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"
    if not source_dir.exists():
        print(f"  Error: Documentation source not found: {source_dir}")
        return False
    try:
        result = subprocess.run(
            ["uv", "run", "sphinx-build", "-M", "html", str(source_dir), str(build_dir)],
            capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT)
        )
        if result.returncode == 0:
            print("  ✓ Documentation built successfully")
            return True
        else:
            print(f"  ✗ Documentation build failed (exit {result.returncode})")
            if result.stderr:
                print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"  ✗ Documentation build failed: {e}")
        return False

def clean_docs():
    build_dir = REPO_ROOT / "docs" / "build"
    if not build_dir.exists():
        print(f"  No build directory found at {build_dir}")
        return
    shutil.rmtree(build_dir)
    print(f"  ✓ Cleaned {build_dir.relative_to(REPO_ROOT)}")

def clean_all():
    targets = [
        REPO_ROOT / "docs" / "build", REPO_ROOT / "docs" / "source" / "example_gallery",
        REPO_ROOT / "docs" / "source" / "generated", REPO_ROOT / "build", REPO_ROOT / "dist",
        REPO_ROOT / ".pytest_cache", REPO_ROOT / ".ruff_cache", REPO_ROOT / ".mypy_cache"
    ]
    cleaned = []
    for t in targets:
        if t.exists():
            try:
                shutil.rmtree(t)
                cleaned.append(str(t.relative_to(REPO_ROOT)))
            except Exception as e:
                print(f"  ✗ Failed to clean {t}: {e}")
    for pycache in REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache)
    print(f"  ✓ Cleaned {len(cleaned)} items" if cleaned else "  Nothing to clean")

def ruff_check():
    try:
        result = subprocess.run(["uv", "run", "ruff", "check", "."], capture_output=True, text=True, cwd=str(REPO_ROOT))
        print(result.stdout)
        if result.returncode == 0:
            print("  ✓ No issues found")
            return True
        print("  ✗ Issues found")
        return False
    except Exception as e:
        print(f"  ✗ ruff check failed: {e}")
        return False

def ruff_format():
    try:
        result = subprocess.run(["uv", "run", "ruff", "format", "."], capture_output=True, text=True, cwd=str(REPO_ROOT))
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"  ✗ ruff format failed: {e}")
        return False

def generate_hpc_pack(solver_name: str) -> Path:
    """
    Run Experiments/<Solver>/generate_pack.sh which should create a 'pack' directory.
    Move the pack to REPO_ROOT/hpc-packs/<solver_name> (creating parent dirs).
    Return the final hpc-pack dir.
    """
    solver_dir = REPO_ROOT / "Experiments" / solver_name
    gen_script = solver_dir / "generate_pack.sh"

    if not gen_script.exists():
        raise FileNotFoundError(f"Missing generate script: {gen_script}")

    # Ensure executable
    gen_script.chmod(gen_script.stat().st_mode | 0o111)

    # Run the script (it should create a local 'pack' subdir)
    print(f"  → Running {gen_script} to generate job pack...")
    result = subprocess.run(["bash", str(gen_script)], capture_output=True, text=True, cwd=str(solver_dir))
    if result.returncode != 0:
        raise RuntimeError(f"generate_pack.sh failed:\n{result.stderr}")

    local_pack = solver_dir / "pack"
    if not local_pack.exists():
        raise FileNotFoundError(f"generate_pack.sh did not produce '{local_pack}'")

    dest = REPO_ROOT / "hpc-packs" / solver_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # move contents of local_pack to dest/pack (keep same layout)
    shutil.move(str(local_pack), str(dest / "pack"))
    print(f"  ✓ Pack created at: {dest}")
    return dest

def submit_hpc_pack(pack_dir: Path):
    """
    Submit all .lsf job files inside pack_dir/pack using bsub.
    If bsub is not available, print the commands (for manual submission).
    """
    pack_files = sorted((pack_dir / "pack").glob("*.lsf"))
    if not pack_files:
        print("  ✗ No .lsf files found in pack to submit.")
        return

    # Try to submit with bsub
    bsub_available = shutil.which("bsub") is not None
    if not bsub_available:
        print("  ⚠ bsub not found in PATH. To submit manually run:")
        for f in pack_files:
            print(f"    bsub < {f}")
        return

    print(f"  → Submitting {len(pack_files)} jobs with bsub ...")
    for f in pack_files:
        result = subprocess.run(["bsub", "<", str(f)], shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    submitted {f.name}")
        else:
            print(f"    failed to submit {f.name}: {result.stderr[:200]}")

def handle_hpc(solver: str):
    print(f"  → Submitting HPC job pack for: {solver}")
    try:
        pack_dir = generate_hpc_pack(solver)
    except Exception as e:
        print(f"  ✗ Failed to generate pack: {e}")
        return
    submit_hpc_pack(pack_dir)
    print(f"  ✓ HPC jobs submitted for {solver}")

def main():
    parser = argparse.ArgumentParser(description="Run example scripts and manage docs")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--build-docs", action="store_true")
    parser.add_argument("--clean-docs", action="store_true")
    parser.add_argument("--clean-all", action="store_true")
    parser.add_argument("--lint", action="store_true")
    parser.add_argument("--format", action="store_true")
    parser.add_argument("--hpc", choices=["spectral", "fv", "all"], help="Generate and submit HPC pack")

    if len(sys.argv) == 1:
        parser.print_help()
        print("\n Error: No arguments provided. Please specify at least one option.\n")
        sys.exit(1)

    args = parser.parse_args()

    if args.clean_all:
        clean_all()
    if args.clean_docs:
        clean_docs()
    if args.lint:
        ruff_check()
    if args.format:
        ruff_format()
    if args.build_docs:
        build_docs()

    if args.hpc:
        if args.hpc == "all":
            handle_hpc("Spectral-Solver")
            handle_hpc("FV_Solver")
        else:
            # map choice to folder names in Experiments
            mapping = {"spectral": "Spectral-Solver", "fv": "FV_Solver"}
            handle_hpc(mapping.get(args.hpc, args.hpc))
        return

    if args.compute or args.plot:
        compute_scripts, plot_scripts = discover_scripts()
        print(f"\nFound {len(compute_scripts)} compute scripts and {len(plot_scripts)} plot scripts")
        if args.compute:
            run_scripts(compute_scripts)
        if args.plot:
            run_scripts(plot_scripts)

if __name__ == "__main__":
    main()
    