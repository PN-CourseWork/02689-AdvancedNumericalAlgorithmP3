"""
MLflow Interaction Utilities for LDC Plotting.

Handles finding runs, downloading artifacts, loading timeseries data,
and uploading plots to MLflow.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def find_matching_run(cfg: DictConfig, tracking_uri: str) -> tuple[str, Optional[str]]:
    """Find MLflow run matching the config parameters.

    Returns
    -------
    tuple[str, Optional[str]]
        (run_id, parent_run_id) - parent_run_id is None if not a sweep child
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    # Build filter string for matching runs
    solver_name = cfg.solver.name
    N = cfg.N
    Re = cfg.Re

    filter_parts = [
        f"params.Re = '{Re}'",
        f"params.nx = '{N}'",
        f"params.ny = '{N}'",
        "attributes.status = 'FINISHED'",
    ]

    # Add solver-specific filter
    if solver_name == "spectral":
        filter_parts.append(f"params.basis_type = '{cfg.solver.basis_type}'")
    elif solver_name == "fv":
        filter_parts.append(
            f"params.convection_scheme = '{cfg.solver.convection_scheme}'"
        )

    filter_string = " AND ".join(filter_parts)

    log.info(f"Searching in experiment: {experiment_name}")
    log.info(f"Filter: solver={solver_name}, N={N}, Re={Re}")

    # Search for runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )

    if not runs:
        raise ValueError(
            f"No matching runs found for solver={solver_name}, N={N}, Re={Re}\n"
            f"Filter used: {filter_string}"
        )

    # Return most recent matching run
    run = runs[0]
    parent_run_id = run.data.tags.get("parent_run_id")

    log.info(f"Found run: {run.info.run_name} (id: {run.info.run_id[:8]}...)")
    if parent_run_id:
        log.info(f"  Parent run: {parent_run_id[:8]}...")

    return run.info.run_id, parent_run_id


def find_sibling_runs(parent_run_id: str, tracking_uri: str) -> list[dict]:
    """Find all child runs of a parent (siblings in a sweep).

    Returns list of dicts with run info for comparison plotting.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get parent run to find experiment
    parent_run = client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id

    # Find all FINISHED children of this parent
    filter_string = (
        f"tags.parent_run_id = '{parent_run_id}' AND attributes.status = 'FINISHED'"
    )

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["params.nx ASC"],  # Sort by N for nice legend order
        max_results=50,
    )

    siblings = []
    for run in runs:
        run_name = run.info.run_name or ""

        # Extract solver name from run_name (format: {solver}_N{n} or {solver}_N{n}_Re{re})
        # Examples: "fv_N32", "spectral_N33", "spectral_fsg_N16"
        if "_N" in run_name:
            solver_name = run_name.rsplit("_N", 1)[
                0
            ]  # rsplit to handle underscores in solver name
        else:
            solver_name = "unknown"

        siblings.append(
            {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "N": int(run.data.params.get("nx", 0)),
                "Re": float(run.data.params.get("Re", 0)),
                "solver": solver_name,
                "status": run.info.status,
            }
        )

    finished = sum(1 for s in siblings if s["status"] == "FINISHED")
    log.info(f"Found {len(siblings)} sibling runs in sweep ({finished} finished)")
    return siblings


def download_mlflow_artifacts(run_id: str, tracking_uri: str) -> Path:
    """Download solution artifacts from MLflow run to temp directory."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    log.info(f"Downloading artifacts from: {run.info.run_name}")

    tmpdir = tempfile.mkdtemp(prefix="ldc_plot_")
    artifact_path = client.download_artifacts(run_id, "", tmpdir)

    return Path(artifact_path)


def load_timeseries_from_mlflow(run_id: str, tracking_uri: str) -> pd.DataFrame:
    """Load timeseries metrics from MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get metric history
    metrics_to_fetch = [
        "residual",
        "u_residual",
        "v_residual",
        "continuity_residual",
        "energy",
        "enstrophy",
        "palinstrophy",
    ]

    data = {}
    for metric_name in metrics_to_fetch:
        try:
            history = client.get_metric_history(run_id, metric_name)
            if history:
                data[metric_name] = [
                    m.value for m in sorted(history, key=lambda x: x.step)
                ]
        except Exception:
            pass  # Metric might not exist

    if not data:
        return pd.DataFrame()

    # Create DataFrame with iteration index
    max_len = max(len(v) for v in data.values())
    df = pd.DataFrame({k: v + [None] * (max_len - len(v)) for k, v in data.items()})
    df["iteration"] = range(len(df))

    return df


def upload_plots_to_mlflow(
    run_id: str, plot_paths: list, tracking_uri: str, artifact_subdir: str = "plots"
):
    """Upload generated plots to MLflow run as artifacts."""
    mlflow.set_tracking_uri(tracking_uri)

    valid_paths = [p for p in plot_paths if p and p.exists()]

    # Check if we're already in an active run
    active_run = mlflow.active_run()
    if active_run and active_run.info.run_id == run_id:
        # Already in the correct run, just log artifacts
        for path in valid_paths:
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
            log.info(f"Uploaded: {artifact_subdir}/{path.name}")
    else:
        # Start/resume run to upload artifacts
        with mlflow.start_run(run_id=run_id, nested=True):
            for path in valid_paths:
                mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
                log.info(f"Uploaded: {artifact_subdir}/{path.name}")
