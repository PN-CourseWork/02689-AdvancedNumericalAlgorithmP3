"""
LDC Plotting Package - Modular structure for lid-driven cavity plots.

This package provides:
- Individual run plots (fields, streamlines, vorticity, centerlines, convergence)
- Comparison plots (Ghia benchmark comparisons)
- MLflow integration for artifact management
- High-level orchestration functions

Main API
--------
generate_plots_for_run : function
    Generate all plots for a single completed run
generate_comparison_plots_for_sweep : function
    Generate comparison plots for sweep results
main : function
    Hydra entry point for standalone CLI usage
"""

from .convergence import plot_convergence
from .data_loading import (
    fields_to_dataframe,
    load_fields_from_zarr,
    restructure_fields,
)
from .fields import plot_fields, plot_streamlines, plot_vorticity
from .mlflow_utils import (
    download_mlflow_artifacts,
    find_matching_run,
    find_sibling_runs,
    load_timeseries_from_mlflow,
    upload_plots_to_mlflow,
)
from .orchestrator import (
    generate_comparison_plots_for_sweep,
    generate_plots_for_run,
    main,
)

# Import style module to trigger sns.set_theme() on package import
from . import style  # noqa: F401

from .validation import plot_centerlines, plot_ghia_comparison

__all__ = [
    # High-level API (most commonly used)
    "generate_plots_for_run",
    "generate_comparison_plots_for_sweep",
    "main",
    # Individual plot functions
    "plot_fields",
    "plot_streamlines",
    "plot_vorticity",
    "plot_centerlines",
    "plot_convergence",
    "plot_ghia_comparison",
    # MLflow utilities
    "find_matching_run",
    "find_sibling_runs",
    "download_mlflow_artifacts",
    "load_timeseries_from_mlflow",
    "upload_plots_to_mlflow",
    # Data utilities
    "load_fields_from_zarr",
    "restructure_fields",
    "fields_to_dataframe",
]
