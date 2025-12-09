"""
LDC Plotting Package.

Provides plot generation for lid-driven cavity solver results.
"""

from .convergence import plot_convergence
from .data_loading import fields_to_dataframe, load_fields_from_zarr, restructure_fields
from .fields import plot_vorticity
from .mlflow_utils import (
    download_mlflow_artifacts,
    find_sibling_runs,
    load_timeseries_from_mlflow,
    upload_plots_to_mlflow,
)
from .orchestrator import generate_comparison_plots_for_sweep, generate_plots_for_run
from .pyvista_fields import generate_pyvista_field_plots
from .validation import plot_ghia_comparison

# Import style module to trigger sns.set_theme() on package import
from . import style  # noqa: F401

__all__ = [
    "generate_plots_for_run",
    "generate_comparison_plots_for_sweep",
    "generate_pyvista_field_plots",
    "plot_vorticity",
    "plot_convergence",
    "plot_ghia_comparison",
    "find_sibling_runs",
    "download_mlflow_artifacts",
    "load_timeseries_from_mlflow",
    "upload_plots_to_mlflow",
    "load_fields_from_zarr",
    "restructure_fields",
    "fields_to_dataframe",
]
