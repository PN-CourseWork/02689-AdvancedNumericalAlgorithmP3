"""
LDC Solver Runner - Hydra + MLflow integration for FV and Spectral solvers.

STANDARD USAGE - Always use -m (multirun mode):
    # Validation experiments
    uv run python run_solver.py -m +experiment=validation/ghia

    # Benchmarking experiments
    uv run python run_solver.py -m +experiment=benchmarking/multigrid_comparison

    # Quick testing
    uv run python run_solver.py -m +experiment=testing/quick_test

    # Custom sweeps
    uv run python run_solver.py -m solver=fv N=16,32,64 Re=100,400

Multirun mode provides:
    - Parent runs for organizing results
    - Automatic plot generation (individual + comparisons)
    - Everything uploaded to MLflow

Single runs (testing only - no plots generated):
    uv run python run_solver.py solver=fv N=32 Re=100
    uv run python run_solver.py solver=spectral/sg N=15 Re=100

Plot generation (separate tool):
    uv run python plot_runs.py +experiment=validation/ghia
    uv run python plot_runs.py parent_run_id=abc123

MLflow modes:
    local  - file-based ./mlruns (default)
    coolify - remote server (requires .env with credentials)
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

    # Check if running in multirun mode
    try:
        import hydra.core.hydra_config
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        is_multirun = hydra_cfg.mode.name == "MULTIRUN"

        # Warn if using experiment config without multirun
        # Check if sweep params are defined in config (indicates experiment config)
        has_sweep_params = "sweeper" in OmegaConf.to_container(cfg.get("hydra", {}), resolve=False)

        if not is_multirun and has_sweep_params:
            log.warning(
                "\n" + "="*80 + "\n"
                "WARNING: You're using an experiment config without multirun mode!\n"
                "Experiment configs are designed for sweeps. Add -m flag:\n"
                "  uv run python run_solver.py -m +experiment=...\n"
                "\nContinuing with single run (no plots will be generated)...\n"
                + "="*80
            )
    except Exception:
        is_multirun = False

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


if __name__ == "__main__":
    main()
