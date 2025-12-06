"""
High-level Plot Generation Orchestration for LDC.

Coordinates the generation of all plots for individual runs and sweep comparisons.
Provides both a direct API for programmatic use and a Hydra entry point for CLI.
"""

import logging
from pathlib import Path
from typing import Optional

import hydra
import mlflow
from dotenv import load_dotenv
from omegaconf import DictConfig

from .convergence import plot_convergence
from .data_loading import fields_to_dataframe, load_fields_from_zarr
from .fields import plot_fields, plot_streamlines, plot_vorticity
from .mlflow_utils import (
    download_mlflow_artifacts,
    find_matching_run,
    find_sibling_runs,
    load_timeseries_from_mlflow,
    upload_plots_to_mlflow,
)
from .validation import plot_centerlines, plot_ghia_comparison

load_dotenv()
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

    Called directly from run_solver.py after solver completes.

    Parameters
    ----------
    run_id : str
        MLflow run ID
    tracking_uri : str
        MLflow tracking URI
    output_dir : Path
        Directory to save plots
    solver_name : str
        Solver name (e.g., "spectral", "spectral_fsg", "fv")
    N : int
        Grid size parameter
    Re : float
        Reynolds number
    parent_run_id : str, optional
        Parent run ID if this is part of a sweep
    upload_to_mlflow : bool
        Whether to upload plots to MLflow

    Returns
    -------
    list[Path]
        List of generated plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download artifacts and load data
    artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
    fields = load_fields_from_zarr(artifact_dir)
    fields_df = fields_to_dataframe(fields)
    timeseries_df = load_timeseries_from_mlflow(run_id, tracking_uri)

    log.info(f"Generating plots for {solver_name} N={N} Re={Re}")

    # Generate individual plots
    plot_paths = []
    plot_paths.append(plot_fields(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_streamlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_vorticity(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_centerlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_convergence(timeseries_df, Re, solver_name, N, output_dir))
    ghia_path = plot_ghia_comparison(
        [
            {
                "run_id": run_id,
                "N": N,
                "Re": Re,
                "solver": solver_name,
                "status": "FINISHED",
            }
        ],
        tracking_uri,
        output_dir,
    )
    if ghia_path:
        plot_paths.append(ghia_path)

    plot_paths = [p for p in plot_paths if p is not None]
    log.info(f"Generated {len(plot_paths)} plots for run")

    # Upload to individual run
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
    """Generate comparison plots for all parent runs after sweep completes.

    Called from MLflow callback's on_multirun_end after all jobs finish.

    Parameters
    ----------
    parent_run_ids : list[str]
        List of parent run IDs (one per Re value)
    tracking_uri : str
        MLflow tracking URI
    output_dir : Path
        Base output directory for comparison plots
    upload_to_mlflow : bool
        Whether to upload plots to MLflow

    Returns
    -------
    dict[str, Path]
        Mapping of parent_run_id to comparison plot path
    """
    mlflow.set_tracking_uri(tracking_uri)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for parent_run_id in parent_run_ids:
        log.info(f"Generating comparison plot for parent run: {parent_run_id[:8]}...")

        # Find all children of this parent
        siblings = find_sibling_runs(parent_run_id, tracking_uri)

        if len(siblings) < 2:
            log.warning(f"  Only {len(siblings)} child run(s), skipping comparison")
            continue

        # Check all siblings are finished
        unfinished = [s for s in siblings if s.get("status") != "FINISHED"]
        if unfinished:
            log.warning(f"  {len(unfinished)} run(s) still not finished, skipping")
            continue

        # Get parent run info for naming
        client = mlflow.tracking.MlflowClient()
        parent_run = client.get_run(parent_run_id)
        parent_name = parent_run.info.run_name or parent_run_id[:8]

        # Create comparison plot
        comparison_dir = output_dir / parent_name
        comparison_dir.mkdir(exist_ok=True)

        comparison_path = plot_ghia_comparison(siblings, tracking_uri, comparison_dir)

        if comparison_path:
            results[parent_run_id] = comparison_path
            log.info(f"  Created comparison plot: {comparison_path.name}")

            if upload_to_mlflow:
                upload_plots_to_mlflow(
                    parent_run_id, [comparison_path], tracking_uri, "plots"
                )
                log.info("  Uploaded to parent run")

    log.info(f"Generated {len(results)} comparison plot(s)")
    return results


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point - finds matching run and generates plots."""

    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    solver_name = cfg.solver.name
    N = cfg.N
    Re = cfg.Re

    # Find matching MLflow run
    run_id, parent_run_id = find_matching_run(cfg, tracking_uri)

    # Setup output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download artifacts for this run
    artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
    fields = load_fields_from_zarr(artifact_dir)
    fields_df = fields_to_dataframe(fields)
    timeseries_df = load_timeseries_from_mlflow(run_id, tracking_uri)

    log.info(f"Generating plots for {solver_name} N={N} Re={Re}")

    # ==========================================================================
    # Individual run plots
    # ==========================================================================
    plot_paths = []

    plot_paths.append(plot_fields(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_streamlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_vorticity(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_centerlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_convergence(timeseries_df, Re, solver_name, N, output_dir))

    plot_paths = [p for p in plot_paths if p is not None]
    log.info(f"Generated {len(plot_paths)} plots for run")

    # Upload to individual run
    if cfg.get("upload_to_mlflow", True):
        upload_plots_to_mlflow(run_id, plot_paths, tracking_uri)

    # ==========================================================================
    # Comparison plot for parent (if this is part of a sweep)
    # ==========================================================================
    if parent_run_id:
        log.info("This run is part of a sweep - generating comparison plot for parent")

        siblings = find_sibling_runs(parent_run_id, tracking_uri)

        if len(siblings) > 1:
            comparison_dir = output_dir / "comparison"
            comparison_dir.mkdir(exist_ok=True)

            comparison_path = plot_ghia_comparison(
                siblings, tracking_uri, comparison_dir
            )

            if comparison_path and cfg.get("upload_to_mlflow", True):
                upload_plots_to_mlflow(
                    parent_run_id, [comparison_path], tracking_uri, "plots"
                )
                log.info("Comparison plot uploaded to parent run")

    log.info("Done!")


if __name__ == "__main__":
    main()
