"""
LDC Solver Runner - Hydra + MLflow integration for FV and Spectral solvers.

Usage:
    # FV solver (32 cells)
    uv run python run_solver.py solver=fv N=32 Re=100 max_iterations=100

    # Spectral solver (N=15 gives 16 nodes)
    uv run python run_solver.py solver=spectral N=15 Re=100 max_iterations=100

    # MLflow modes:
    #   local-files  - file-based ./mlruns (default)
    #   coolify      - remote server (requires .env with credentials)
    uv run python run_solver.py solver=fv mlflow=local-files
    uv run python run_solver.py solver=fv mlflow=coolify

    # Parameter sweep
    uv run python run_solver.py -m solver=fv N=16,32,64 Re=100,400

Setup for remote MLflow:
    cp .env.template .env
    # Edit .env with your credentials
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import mlflow
from dotenv import load_dotenv
from hydra.utils import instantiate
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

# Load .env file (for MLflow credentials)
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

log = logging.getLogger(__name__)


# =============================================================================
# Solver Factory
# =============================================================================


def create_solver(cfg: DictConfig):
    """Instantiate solver using Hydra's instantiate on solver subtree."""
    return instantiate(cfg.solver, _convert_="partial")


# =============================================================================
# MLflow Logging
# =============================================================================


def setup_mlflow(cfg: DictConfig) -> str:
    """Setup MLflow tracking and return experiment name."""
    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    # If defaulting to local file backend, clear any env override
    if str(cfg.mlflow.get("mode", "")).lower() in ("files", "local"):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ["MLFLOW_TRACKING_URI"] = str(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    # Build experiment name with optional project prefix
    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        # If experiment was previously deleted, fall back to a new name
        fallback = f"{experiment_name}-restored"
        log.warning(
            "MLflow set_experiment failed for '%s' (%s); falling back to '%s'",
            experiment_name,
            exc,
            fallback,
        )
        experiment_name = fallback
        mlflow.set_experiment(experiment_name)
    return experiment_name


def log_params(solver):
    """Log solver params to MLflow using dataclass to_mlflow method."""
    mlflow.log_params(solver.params.to_mlflow())


def log_metrics_and_timeseries(solver, run_id: str):
    """Log final metrics and timeseries to MLflow."""
    # Final metrics (using dataclass to_mlflow method)
    mlflow.log_metrics(solver.metrics.to_mlflow())

    # Timeseries (batch logging using dataclass to_mlflow_batch method)
    if solver.time_series is not None:
        batch_metrics = solver.time_series.to_mlflow_batch()
        if batch_metrics:
            MlflowClient().log_batch(run_id=run_id, metrics=batch_metrics)


def log_fields(solver):
    """Save solution fields as zarr arrays to MLflow artifacts."""
    import tempfile

    import zarr

    fields = solver.fields

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save each field as separate zarr array
        for name in ["x", "y", "u", "v", "p"]:
            arr = getattr(fields, name)
            zarr_path = Path(tmpdir) / f"{name}.zarr"
            zarr.save(zarr_path, arr)
            mlflow.log_artifact(str(zarr_path), artifact_path="fields")

        log.info("Logged fields: x, y, u, v, p (zarr)")


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point - runs solver with MLflow tracking."""
    log.info(f"Solver: {cfg.solver.name}, N={cfg.N}, Re={cfg.Re}")

    # Setup MLflow
    experiment_name = setup_mlflow(cfg)
    log.info(f"MLflow experiment: {experiment_name}")

    # Create solver
    solver = create_solver(cfg)

    # Build run name (Re is in parent run, not child run name)
    solver_name = cfg.solver.name
    if solver_name.startswith("spectral"):
        run_name = f"{solver_name}_N{cfg.N + 1}"
    else:
        run_name = f"{solver_name}_N{cfg.N}"

    # Check for parent run (from sweep callback)
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")

    # Run with MLflow tracking
    # Use nested=True when parent run is active (same process in local multirun)
    run_tags = {"solver": solver_name}  # Always tag with solver name
    nested = False
    if parent_run_id:
        run_tags["mlflow.parentRunId"] = parent_run_id
        run_tags["parent_run_id"] = (
            parent_run_id  # Also store as regular tag for querying
        )
        run_tags["sweep"] = "child"
        nested = True  # Required when parent run is active in same process

    with mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested) as run:
        log_params(solver)

        # Log Hydra config as artifact
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

        # Tag with HPC job info if available
        job_id = os.environ.get("LSB_JOBID")
        if job_id:
            mlflow.set_tag("lsf.job_id", job_id)
            mlflow.set_tag("lsf.job_name", os.environ.get("LSB_JOBNAME", ""))

        # Solve
        log.info("Starting solver...")
        solver.solve()

        # Log results
        log_metrics_and_timeseries(solver, run.info.run_id)
        log_fields(solver)

        log.info(
            f"Done: {solver.metrics.iterations} iter, "
            f"converged={solver.metrics.converged}, "
            f"time={solver.metrics.wall_time_seconds:.2f}s"
        )

        # Generate plots
        if cfg.get("generate_plots", True):
            from shared.plotting.ldc_plotter import generate_plots_for_run

            output_dir = Path(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
            tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
            generate_plots_for_run(
                run_id=run.info.run_id,
                tracking_uri=tracking_uri,
                output_dir=output_dir / "plots",
                solver_name=solver_name,
                N=cfg.N,
                Re=cfg.Re,
                parent_run_id=parent_run_id,
                upload_to_mlflow=True,
            )


if __name__ == "__main__":
    main()
