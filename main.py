#!/usr/bin/env python3
"""Project CLI - run with `uv run python main.py` for interactive mode."""

import argparse
import sys

from cli import (
    fetch_mlflow,
    run_scripts,
    build_docs,
    clean_all,
    ruff_check,
    ruff_format,
    hpc_submit,
    interactive,
)


def main():
    parser = argparse.ArgumentParser(description="ANA-P3 Project Manager")
    parser.add_argument("--fetch", action="store_true", help="Fetch MLflow artifacts")
    parser.add_argument("--compute", action="store_true", help="Run compute scripts")
    parser.add_argument("--plot", action="store_true", help="Run plot scripts")
    parser.add_argument("--build-docs", action="store_true", help="Build documentation")
    parser.add_argument("--clean", action="store_true", help="Clean caches")
    parser.add_argument("--lint", action="store_true", help="Run ruff linter")
    parser.add_argument("--format", action="store_true", help="Format with ruff")
    parser.add_argument("--hpc", choices=["spectral", "fv", "all"], help="Submit HPC jobs")
    parser.add_argument("--dry-run", action="store_true", help="HPC dry run")

    args = parser.parse_args()

    # No args = interactive mode
    if len(sys.argv) == 1:
        try:
            interactive()
        except KeyboardInterrupt:
            print("\n")
        return

    # CLI mode
    if args.clean:
        clean_all()
    if args.lint:
        ruff_check()
    if args.format:
        ruff_format()
    if args.fetch:
        fetch_mlflow()
    if args.build_docs:
        build_docs()
    if args.compute:
        run_scripts("compute")
    if args.plot:
        run_scripts("plot")
    if args.hpc:
        hpc_submit(args.hpc, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
