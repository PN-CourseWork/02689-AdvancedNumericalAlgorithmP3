"""
Plot generation script for LDC experiments.

Finds parent runs for an experiment and generates:
1. Individual plots for all child runs
2. Comparison plots for each parent run

Usage:
    # Plot all runs in an experiment
    uv run python plot_runs.py experiment_name=LDC-Validation

    # Plot runs for a specific parent run ID
    uv run python plot_runs.py parent_run_id=abc123

    # Plot using experiment config
    uv run python plot_runs.py +experiment=fv_validation

    # Plot with multirun (regenerate plots for sweep)
    uv run python plot_runs.py -m +experiment=fv_validation
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import mlflow
from dotenv import load_dotenv
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()
log = logging.getLogger(__name__)


def find_parent_runs_for_experiment(
    experiment_name: str, tracking_uri: str
) -> list[dict]:
    """Find all parent runs for an experiment.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name
    tracking_uri : str
        MLflow tracking URI

    Returns
    -------
    list[dict]
        List of parent run info dicts with run_id, name, Re (if tagged)
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Search for parent runs
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.sweep = 'parent'",
        order_by=["start_time DESC"],
    )

    if runs.empty:
        log.warning(f"No parent runs found in experiment: {experiment_name}")
        return []

    parent_runs = []
    for _, row in runs.iterrows():
        parent_info = {
            "run_id": row["run_id"],
            "name": row["tags.mlflow.runName"],
        }
        # Extract Re if tagged
        if "tags.Re" in row and row["tags.Re"]:
            parent_info["Re"] = int(row["tags.Re"])

        parent_runs.append(parent_info)

    log.info(f"Found {len(parent_runs)} parent run(s) in {experiment_name}")
    return parent_runs


def plot_all_runs_for_parent(
    parent_run_id: str,
    tracking_uri: str,
    output_dir: Path,
    upload_to_mlflow: bool = True,
) -> dict:
    """Generate all plots for a parent run and its children.

    Parameters
    ----------
    parent_run_id : str
        Parent run ID
    tracking_uri : str
        MLflow tracking URI
    output_dir : Path
        Output directory for plots
    upload_to_mlflow : bool
        Whether to upload plots to MLflow

    Returns
    -------
    dict
        Summary with child_plots (list) and comparison_plot (Path or None)
    """
    from shared.plotting.ldc import (
        find_sibling_runs,
        generate_plots_for_run,
        plot_ghia_comparison,
        upload_plots_to_mlflow,
    )

    mlflow.set_tracking_uri(tracking_uri)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get parent run info
    client = mlflow.tracking.MlflowClient()
    parent_run = client.get_run(parent_run_id)
    parent_name = parent_run.info.run_name or parent_run_id[:8]

    log.info(f"Generating plots for parent run: {parent_name} ({parent_run_id[:8]})")

    # Find all child runs
    siblings = find_sibling_runs(parent_run_id, tracking_uri)

    if not siblings:
        log.warning(f"  No child runs found for parent: {parent_name}")
        return {"child_plots": [], "comparison_plot": None}

    log.info(f"  Found {len(siblings)} child run(s)")

    # Filter to finished runs
    finished_siblings = [s for s in siblings if s.get("status") == "FINISHED"]
    if len(finished_siblings) < len(siblings):
        log.warning(
            f"  Only {len(finished_siblings)}/{len(siblings)} runs finished, "
            "plotting finished runs only"
        )

    # Generate individual plots for each child
    child_plot_results = []
    for i, sibling in enumerate(finished_siblings, 1):
        run_id = sibling["run_id"]
        solver = sibling.get("solver", "unknown")
        N = sibling["N"]
        Re = sibling["Re"]

        log.info(f"  [{i}/{len(finished_siblings)}] Plotting {solver} N={N} Re={Re}")

        child_output_dir = output_dir / parent_name / f"{solver}_N{N}_Re{Re}"
        child_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_paths = generate_plots_for_run(
                run_id=run_id,
                tracking_uri=tracking_uri,
                output_dir=child_output_dir,
                solver_name=solver,
                N=N,
                Re=Re,
                parent_run_id=parent_run_id,
                upload_to_mlflow=upload_to_mlflow,
            )
            child_plot_results.append(
                {"run_id": run_id, "plots": plot_paths, "status": "success"}
            )
            log.info(f"    Generated {len(plot_paths)} plot(s)")
        except Exception as e:
            log.error(f"    Failed to generate plots for {run_id}: {e}")
            child_plot_results.append({"run_id": run_id, "status": "failed", "error": str(e)})

    # Generate comparison plot for parent
    comparison_plot = None
    if len(finished_siblings) >= 2:
        log.info(f"  Generating comparison plot for parent: {parent_name}")
        comparison_dir = output_dir / parent_name / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        try:
            comparison_plot = plot_ghia_comparison(
                finished_siblings, tracking_uri, comparison_dir
            )

            if comparison_plot and upload_to_mlflow:
                upload_plots_to_mlflow(
                    parent_run_id, [comparison_plot], tracking_uri, "plots"
                )
                log.info(f"    Uploaded comparison plot to parent run")
        except Exception as e:
            log.error(f"    Failed to generate comparison plot: {e}")
    else:
        log.warning(
            f"  Only {len(finished_siblings)} finished run(s), "
            "skipping comparison plot (need at least 2)"
        )

    summary = {
        "parent_run_id": parent_run_id,
        "parent_name": parent_name,
        "child_plots": child_plot_results,
        "comparison_plot": comparison_plot,
    }

    log.info(
        f"Completed plotting for {parent_name}: "
        f"{len([r for r in child_plot_results if r['status'] == 'success'])} child runs, "
        f"comparison={'yes' if comparison_plot else 'no'}"
    )

    return summary


def plot_experiment(
    experiment_name: str,
    tracking_uri: str,
    output_dir: Path,
    parent_run_ids: Optional[list[str]] = None,
    upload_to_mlflow: bool = True,
) -> list[dict]:
    """Generate all plots for an experiment.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name
    tracking_uri : str
        MLflow tracking URI
    output_dir : Path
        Output directory for plots
    parent_run_ids : list[str], optional
        Specific parent run IDs to plot (if None, plots all parents in experiment)
    upload_to_mlflow : bool
        Whether to upload plots to MLflow

    Returns
    -------
    list[dict]
        List of summaries for each parent run
    """
    mlflow.set_tracking_uri(tracking_uri)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find parent runs if not provided
    if parent_run_ids is None:
        parent_runs = find_parent_runs_for_experiment(experiment_name, tracking_uri)
        parent_run_ids = [p["run_id"] for p in parent_runs]

    if not parent_run_ids:
        log.warning("No parent runs to plot")
        return []

    log.info(f"Generating plots for {len(parent_run_ids)} parent run(s)")

    # Plot each parent run
    summaries = []
    for i, parent_run_id in enumerate(parent_run_ids, 1):
        log.info(f"[{i}/{len(parent_run_ids)}] Processing parent run {parent_run_id[:8]}")

        summary = plot_all_runs_for_parent(
            parent_run_id=parent_run_id,
            tracking_uri=tracking_uri,
            output_dir=output_dir,
            upload_to_mlflow=upload_to_mlflow,
        )
        summaries.append(summary)

    # Print summary
    log.info("\n" + "=" * 80)
    log.info("PLOTTING SUMMARY")
    log.info("=" * 80)
    for summary in summaries:
        success_count = len([r for r in summary["child_plots"] if r["status"] == "success"])
        total_count = len(summary["child_plots"])
        log.info(
            f"{summary['parent_name']}: {success_count}/{total_count} child runs plotted, "
            f"comparison={'yes' if summary['comparison_plot'] else 'no'}"
        )

    return summaries


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for plot generation.

    Supports multiple modes:
    1. Plot by experiment name: experiment_name=LDC-Validation
    2. Plot by parent run ID: parent_run_id=abc123
    3. Plot using experiment config: +experiment=fv_validation
    """
    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")

    # Determine output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Mode 1: Explicit parent_run_id provided
    if cfg.get("parent_run_id"):
        parent_run_id = cfg.parent_run_id
        log.info(f"Plotting for parent run: {parent_run_id}")

        plot_all_runs_for_parent(
            parent_run_id=parent_run_id,
            tracking_uri=tracking_uri,
            output_dir=output_dir,
            upload_to_mlflow=cfg.get("upload_to_mlflow", True),
        )
        return

    # Mode 2: Use experiment_name from config
    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        full_experiment_name = f"{project_prefix}/{experiment_name}"
    else:
        full_experiment_name = experiment_name

    log.info(f"Plotting for experiment: {full_experiment_name}")

    # Check if specific parent_run_ids provided as list
    parent_run_ids = cfg.get("parent_run_ids", None)

    plot_experiment(
        experiment_name=full_experiment_name,
        tracking_uri=tracking_uri,
        output_dir=output_dir,
        parent_run_ids=parent_run_ids,
        upload_to_mlflow=cfg.get("upload_to_mlflow", True),
    )


if __name__ == "__main__":
    main()
