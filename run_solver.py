"""
LDC Solver Runner - Hydra + MLflow integration for FV and Spectral solvers.

Usage:
    # FV solver (32 cells)
    uv run python run_solver.py solver=fv N=32 Re=100 max_iterations=100

    # Spectral solver (N=15 gives 16 nodes)
    uv run python run_solver.py solver=spectral N=15 Re=100 max_iterations=100

    # MLflow modes:
    #   local-files  - file-based ./mlruns (default, no docker)
    #   local-docker - local docker-compose (cd mlflow-server && docker compose up -d)
    #   coolify      - remote server (requires .env with credentials)
    uv run python run_solver.py solver=fv mlflow=local-docker
    uv run python run_solver.py solver=fv mlflow=coolify

    # Parameter sweep
    uv run python run_solver.py -m solver=fv N=16,32,64 Re=100,400

Setup for remote MLflow:
    cp .env.template .env
    # Edit .env with your credentials
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file (for MLflow credentials)
load_dotenv()

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# =============================================================================
# Solver Factory
# =============================================================================


def create_solver(cfg: DictConfig):
    """Create solver instance from Hydra config."""
    solver_name = cfg.solver.name

    # Common parameters
    common = {
        "Re": cfg.Re,
        "lid_velocity": cfg.lid_velocity,
        "Lx": cfg.Lx,
        "Ly": cfg.Ly,
        "nx": cfg.N,
        "ny": cfg.N,
        "max_iterations": cfg.max_iterations,
        "tolerance": cfg.tolerance,
    }

    if solver_name == "fv":
        from solvers import FVSolver

        return FVSolver(
            **common,
            convection_scheme=cfg.solver.convection_scheme,
            limiter=cfg.solver.limiter,
            alpha_uv=cfg.solver.alpha_uv,
            alpha_p=cfg.solver.alpha_p,
            linear_solver_tol=cfg.solver.linear_solver_tol,
        )
    elif solver_name == "spectral":
        from solvers import SpectralSolver

        return SpectralSolver(
            **common,
            basis_type=cfg.solver.basis_type,
            CFL=cfg.solver.CFL,
            beta_squared=cfg.solver.beta_squared,
            corner_smoothing=cfg.solver.corner_smoothing,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


# =============================================================================
# MLflow Logging
# =============================================================================


def setup_mlflow(cfg: DictConfig) -> str:
    """Setup MLflow tracking and return experiment name."""
    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Build experiment name with optional project prefix
    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"

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


def save_and_log_artifact(solver, cfg: DictConfig, output_dir: str):
    """Save solver output to HDF5 and log as MLflow artifact."""
    # Determine output filename
    solver_name = cfg.solver.name.upper()
    if solver_name == "SPECTRAL":
        n_nodes = cfg.N + 1
        filename = f"LDC_{solver_name}_N{n_nodes}_Re{int(cfg.Re)}.h5"
    else:
        filename = f"LDC_{solver_name}_N{cfg.N}_Re{int(cfg.Re)}.h5"

    output_path = Path(output_dir) / filename
    solver.save(output_path)
    log.info(f"Saved results to: {output_path}")

    mlflow.log_artifact(str(output_path))


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point - runs solver with MLflow tracking."""
    output_dir = HydraConfig.get().runtime.output_dir

    # Log resolved config
    log.info(f"Solver: {cfg.solver.name}, N={cfg.N}, Re={cfg.Re}")
    log.info(f"Output dir: {output_dir}")

    # Save resolved config
    OmegaConf.save(cfg, Path(output_dir) / "config.yaml")

    # Setup MLflow
    experiment_name = setup_mlflow(cfg)
    log.info(f"MLflow experiment: {experiment_name}")

    # Create solver
    solver = create_solver(cfg)

    # Build run name
    solver_name = cfg.solver.name
    if solver_name == "spectral":
        run_name = f"{solver_name}_N{cfg.N + 1}_Re{int(cfg.Re)}"
    else:
        run_name = f"{solver_name}_N{cfg.N}_Re{int(cfg.Re)}"

    # Run with MLflow tracking
    with mlflow.start_run(run_name=run_name) as run:
        log_params(solver)

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
        save_and_log_artifact(solver, cfg, output_dir)

        log.info(
            f"Done: {solver.metrics.iterations} iter, "
            f"converged={solver.metrics.converged}, "
            f"time={solver.metrics.wall_time_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
