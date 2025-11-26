"""Utility modules for plotting and project management."""

from pathlib import Path
from . import plotting
from .ldc_plotter import LDCPlotter
from .ghia_validator import GhiaValidator, plot_validation
from .field_interpolator import UnifiedFieldInterpolator
from .data_io import (
    load_run_data,
    load_fields,
    load_metadata,
    load_multiple_runs,
)

__all__ = [
    "plotting",
    "get_project_root",
    "LDCPlotter",
    "GhiaValidator",
    "plot_validation",
    "UnifiedFieldInterpolator",
    "load_run_data",
    "load_fields",
    "load_metadata",
    "load_multiple_runs",
]


def get_project_root() -> Path:
    """Get project root directory.

    Returns
    -------
    Path
        Project root directory (contains pyproject.toml).
    """
    # Start from this file and search upward for pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback: assume standard structure
    return Path(__file__).resolve().parent.parent.parent
