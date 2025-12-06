"""
LDC Solver Runner - Hydra + MLflow integration for FV and Spectral solvers.

RECOMMENDED USAGE - Parameter sweeps (multirun mode):
    # Use experiment configs for validation sweeps
    uv run python run_solver.py -m +experiment=fv_validation
    uv run python run_solver.py -m +experiment=spectral_validation

    # Custom sweeps
    uv run python run_solver.py -m solver=fv N=16,32,64 Re=100,400

    Multirun mode automatically:
    - Creates parent runs for organizing results
    - Generates all plots (individual + comparisons) at the end
    - Uploads everything to MLflow

Single runs (for testing):
    # FV solver
    uv run python run_solver.py solver=fv N=32 Re=100

    # Spectral solver
    uv run python run_solver.py solver=spectral N=15 Re=100

Plot generation:
    # Regenerate plots for existing experiment (no solver execution)
    uv run python plot_runs.py +experiment=fv_validation

    # Regenerate plots for specific parent run
    uv run python plot_runs.py parent_run_id=abc123

MLflow modes:
    local-files  - file-based ./mlruns (default)
    coolify      - remote server (requires .env with credentials)

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
    """Instantiate solver using Hydra's instantiate on solver subtree.

    Common parameters from root config are passed to the solver constructor.
    """
    return instantiate(
        cfg.solver,
        Re=cfg.Re,
        lid_velocity=cfg.lid_velocity,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        nx=cfg.N,
        ny=cfg.N,
        max_iterations=cfg.max_iterations,
        tolerance=cfg.tolerance,
        _convert_="partial",
    )


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

    # Plot-only mode: regenerate plots without running solver
    # NOTE: For multirun plot regeneration, use plot_runs.py instead!
    if cfg.get("plot_only", False):
        log.warning(
            "plot_only mode is deprecated for run_solver.py. "
            "Use plot_runs.py instead:\n"
            "  uv run python plot_runs.py +experiment=fv_validation"
        )
        log.info("Redirecting to plot_runs.py...")

        # Import and call plot_experiment directly
        sys.path.insert(0, str(Path(__file__).parent))
        from plot_runs import plot_experiment

        tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")

        # Build experiment name
        experiment_name = cfg.experiment_name
        project_prefix = cfg.mlflow.get("project_prefix", "")
        if project_prefix and not experiment_name.startswith("/"):
            experiment_name = f"{project_prefix}/{experiment_name}"

        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )

        plot_experiment(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            output_dir=output_dir,
            upload_to_mlflow=cfg.get("upload_to_mlflow", True),
        )
        log.info("Plot regeneration complete!")
        return

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

        # Plot generation is handled centrally by plot_runs.py
        # - For multirun: plots generated automatically at end via MLflowSweepCallback
        # - For single runs: use `uv run python plot_runs.py +experiment=...`
        # - Or manually: `uv run python plot_runs.py parent_run_id=...`


if __name__ == "__main__":
    main()
