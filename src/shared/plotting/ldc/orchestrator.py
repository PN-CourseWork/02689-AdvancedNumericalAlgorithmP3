"""
Plot Generation for LDC runs.

Provides functions to generate plots for individual runs and sweep comparisons.
"""

import logging
from pathlib import Path
from typing import Optional

import mlflow

from .convergence import plot_convergence
from .data_loading import fields_to_dataframe, load_fields_from_zarr
from .fields import plot_vorticity
from .mlflow_utils import (
    download_mlflow_artifacts,
    find_sibling_runs,
    load_timeseries_from_mlflow,
    upload_plots_to_mlflow,
)
from .pyvista_fields import generate_pyvista_field_plots
from .validation import plot_ghia_comparison

log = logging.getLogger(__name__)


def generate_plots_for_run(
    run_id: str,
    tracking_uri: str,
    output_dir: Path,
    solver_name: str,
    N: int,
    Re: float,
    parent_run_id: Optional[str] = None,
    upload_to_mlflow: bool = True,
) -> list[Path]:
    """Generate all plots for a completed run.

    Artifacts generated:
    - convergence.pdf (matplotlib)
    - ghia_comparison.pdf (matplotlib)
    - vorticity.pdf (matplotlib)
    - u.png, v.png, pressure.png, vel-mag.png, streamlines.png (PyVista)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download artifacts and load data
    artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
    fields = load_fields_from_zarr(artifact_dir)
    fields_df = fields_to_dataframe(fields)
    timeseries_df = load_timeseries_from_mlflow(run_id, tracking_uri)

    log.info(f"Generating plots for {solver_name} N={N} Re={Re}")

    plot_paths = []

    # Matplotlib plots: convergence, ghia comparison, vorticity
    plot_paths.append(plot_convergence(timeseries_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_vorticity(fields_df, Re, solver_name, N, output_dir))

    ghia_path = plot_ghia_comparison(
        [{"run_id": run_id, "N": N, "Re": Re, "solver": solver_name, "status": "FINISHED"}],
        tracking_uri,
        output_dir,
    )
    if ghia_path:
        plot_paths.append(ghia_path)

    # PyVista plots: u, v, pressure, vel-mag, streamlines
    vts_path = artifact_dir / "solution.vts"
    if vts_path.exists():
        pyvista_paths = generate_pyvista_field_plots(vts_path, output_dir)
        plot_paths.extend(pyvista_paths.values())
    else:
        log.warning(f"VTS file not found at {vts_path}, skipping PyVista plots")

    plot_paths = [p for p in plot_paths if p is not None]
    log.info(f"Generated {len(plot_paths)} plots for run")

    if upload_to_mlflow:
        upload_plots_to_mlflow(run_id, plot_paths, tracking_uri)

    log.info("Plotting done!")
    return plot_paths


def generate_comparison_plots_for_sweep(
    parent_run_ids: list[str],
    tracking_uri: str,
    output_dir: Path,
    upload_to_mlflow: bool = True,
) -> dict[str, Path]:
    """Generate comparison plots for all parent runs after sweep completes."""
    mlflow.set_tracking_uri(tracking_uri)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for parent_run_id in parent_run_ids:
        log.info(f"Generating comparison plot for parent run: {parent_run_id[:8]}...")

        siblings = find_sibling_runs(parent_run_id, tracking_uri)

        if len(siblings) < 2:
            log.warning(f"  Only {len(siblings)} child run(s), skipping comparison")
            continue

        unfinished = [s for s in siblings if s.get("status") != "FINISHED"]
        if unfinished:
            log.warning(f"  {len(unfinished)} run(s) still not finished, skipping")
            continue

        client = mlflow.tracking.MlflowClient()
        parent_run = client.get_run(parent_run_id)
        parent_name = parent_run.info.run_name or parent_run_id[:8]

        comparison_dir = output_dir / parent_name
        comparison_dir.mkdir(exist_ok=True)

        comparison_path = plot_ghia_comparison(siblings, tracking_uri, comparison_dir)

        if comparison_path:
            results[parent_run_id] = comparison_path
            log.info(f"  Created comparison plot: {comparison_path.name}")

            if upload_to_mlflow:
                upload_plots_to_mlflow(parent_run_id, [comparison_path], tracking_uri)
                log.info("  Uploaded to parent run")

    log.info(f"Generated {len(results)} comparison plot(s)")
    return results
