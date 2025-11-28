#!/usr/bin/env python3
"""Project CLI - run `uv run python main.py` for interactive mode."""

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

# CLI flags for automation (CI/scripts)
FLAGS = {
    "--fetch": fetch_mlflow,
    "--compute": lambda: run_scripts("compute"),
    "--plot": lambda: run_scripts("plot"),
    "--build-docs": build_docs,
    "--clean": clean_all,
    "--lint": ruff_check,
    "--format": ruff_format,
}

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Interactive mode
        try:
            interactive()
        except KeyboardInterrupt:
            print()
    else:
        # CLI mode
        for flag, action in FLAGS.items():
            if flag in args:
                action()

        if "--hpc" in args:
            idx = args.index("--hpc")
            experiment = (
                args[idx + 1]
                if idx + 1 < len(args) and not args[idx + 1].startswith("--")
                else "all"
            )
            dry_run = "--dry-run" in args
            hpc_submit(experiment, dry_run)
