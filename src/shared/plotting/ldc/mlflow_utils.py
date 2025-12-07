"""
MLflow utilities for LDC plotting.

Handles downloading artifacts, loading timeseries, and uploading plots.
"""

import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

log = logging.getLogger(__name__)


def find_sibling_runs(parent_run_id: str, tracking_uri: str) -> list[dict]:
    """Find all child runs of a parent (siblings in a sweep)."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    parent_run = client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id

    filter_string = (
        f"tags.parent_run_id = '{parent_run_id}' AND attributes.status = 'FINISHED'"
    )

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["params.nx ASC"],
        max_results=50,
    )

    siblings = []
    for run in runs:
        run_name = run.info.run_name or ""
        solver_name = run_name.rsplit("_N", 1)[0] if "_N" in run_name else "unknown"

        siblings.append({
            "run_id": run.info.run_id,
            "run_name": run_name,
            "N": int(run.data.params.get("nx", 0)),
            "Re": float(run.data.params.get("Re", 0)),
            "solver": solver_name,
            "status": run.info.status,
        })

    log.info(f"Found {len(siblings)} sibling runs in sweep")
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

    metrics_to_fetch = [
        "residual", "u_residual", "v_residual", "continuity_residual",
        "energy", "enstrophy", "palinstrophy",
    ]

    data = {}
    for metric_name in metrics_to_fetch:
        try:
            history = client.get_metric_history(run_id, metric_name)
            if history:
                data[metric_name] = [m.value for m in sorted(history, key=lambda x: x.step)]
        except Exception:
            pass

    if not data:
        return pd.DataFrame()

    max_len = max(len(v) for v in data.values())
    df = pd.DataFrame({k: v + [None] * (max_len - len(v)) for k, v in data.items()})
    df["iteration"] = range(len(df))

    return df


def upload_plots_to_mlflow(run_id: str, plot_paths: list, tracking_uri: str):
    """Upload generated plots to MLflow run as artifacts."""
    mlflow.set_tracking_uri(tracking_uri)

    valid_paths = [p for p in plot_paths if p and p.exists()]

    active_run = mlflow.active_run()
    if active_run and active_run.info.run_id == run_id:
        for path in valid_paths:
            mlflow.log_artifact(str(path))
            log.info(f"Uploaded: {path.name}")
    else:
        with mlflow.start_run(run_id=run_id, nested=True):
            for path in valid_paths:
                mlflow.log_artifact(str(path))
                log.info(f"Uploaded: {path.name}")
