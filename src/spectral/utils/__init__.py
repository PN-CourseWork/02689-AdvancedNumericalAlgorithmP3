"""Compatibility wrappers for migrated utilities."""

from utilities.io import ensure_output_dir, load_simulation_data, save_simulation_data
from solvers.metrics import (
    extract_metadata,
    format_dt_latex,
    format_parameter_range,
    build_parameter_string,
    discrete_l2_error,
    discrete_l2_norm,
    discrete_linf_error,
)
from shared.plotting.plotting import get_repo_root

__all__ = [
    # I/O
    "ensure_output_dir",
    "load_simulation_data",
    "save_simulation_data",
    # Formatting
    "extract_metadata",
    "format_dt_latex",
    "format_parameter_range",
    "build_parameter_string",
    # Plotting
    "get_repo_root",
    # Norms
    "discrete_l2_error",
    "discrete_l2_norm",
    "discrete_linf_error",
]
