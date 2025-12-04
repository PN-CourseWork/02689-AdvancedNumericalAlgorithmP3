"""
LDC Solver Runner - Hydra + MLflow integration for FV and Spectral solvers.

Usage:
    # FV solver (32 cells)
    uv run python run_solver.py solver=fv N=32 Re=100 max_iterations=100

    # Spectral solver (N=15 gives 16 nodes)
    uv run python run_solver.py solver=spectral N=15 Re=100 max_iterations=100

    # With Databricks MLflow
    uv run python run_solver.py solver=fv mlflow=databricks

    # Parameter sweep
    uv run python run_solver.py -m solver=fv N=16,32,64 Re=100,400
"""

import logging
import os
from dataclasses import asdict
from pathlib import Path

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import Metric
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

    # Databricks login if needed
    if cfg.mlflow.mode == "databricks":
        mlflow.login()

    mlflow.set_experiment(experiment_name)
    return experiment_name


def log_config_params(cfg: DictConfig):
    """Log flattened config as MLflow parameters."""
    # Common params
    mlflow.log_params(
        {
            "solver": cfg.solver.name,
            "N": cfg.N,
            "Re": cfg.Re,
            "lid_velocity": cfg.lid_velocity,
            "Lx": cfg.Lx,
            "Ly": cfg.Ly,
            "tolerance": cfg.tolerance,
            "max_iterations": cfg.max_iterations,
        }
    )

    # Solver-specific params (flatten solver config)
    solver_params = {
        f"solver.{k}": v for k, v in OmegaConf.to_container(cfg.solver).items() if k != "name"
    }
    mlflow.log_params(solver_params)


def log_metrics_and_timeseries(solver, run_id: str):
    """Log final metrics and timeseries to MLflow."""
    # Final metrics
    metrics_dict = asdict(solver.metrics)
    # Convert booleans to int for MLflow
    metrics_dict = {k: (int(v) if isinstance(v, bool) else v) for k, v in metrics_dict.items()}
    mlflow.log_metrics(metrics_dict)

    # Timeseries (batch logging for efficiency)
    if solver.time_series is not None:
        ts = asdict(solver.time_series)
        batch_metrics = []
        for key, values in ts.items():
            if values is not None:
                for step, value in enumerate(values):
                    if value is not None:
                        batch_metrics.append(Metric(key=key, value=value, timestamp=0, step=step))

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
        log_config_params(cfg)

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
