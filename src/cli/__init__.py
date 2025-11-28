"""CLI module for ANA-P3 project management."""

from .actions import (
    fetch_mlflow,
    run_scripts,
    build_docs,
    clean_all,
    ruff_check,
    ruff_format,
    hpc_submit,
    hpc_status,
    hpc_monitor,
    hpc_kill,
)
from .interactive import interactive

__all__ = [
    "fetch_mlflow",
    "run_scripts",
    "build_docs",
    "clean_all",
    "ruff_check",
    "ruff_format",
    "hpc_submit",
    "hpc_status",
    "hpc_monitor",
    "hpc_kill",
    "interactive",
]
