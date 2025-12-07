"""
LDC Solver - Unified entry point for solving and plotting.

Usage:
    # Run solver + generate plots (default)
    uv run python main.py -m +experiment/validation/ghia=fv

    # Regenerate plots only (no solving)
    uv run python main.py -m +experiment/validation/ghia=fv plot_only=true

    # Single run (testing)
    uv run python main.py solver=fv N=32 Re=100

    # MLflow UI
    uv run mlflow ui
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import hydra
import mlflow
from dotenv import load_dotenv
from hydra.utils import instantiate
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

log = logging.getLogger(__name__)


def create_solver(cfg: DictConfig):
    """Instantiate solver from config."""
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


def setup_mlflow(cfg: DictConfig) -> str:
    """Setup MLflow tracking and return experiment name."""
    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    if str(cfg.mlflow.get("mode", "")).lower() in ("files", "local"):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ["MLFLOW_TRACKING_URI"] = str(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        fallback = f"{experiment_name}-restored"
        log.warning(f"MLflow set_experiment failed ({exc}); using '{fallback}'")
        experiment_name = fallback
        mlflow.set_experiment(experiment_name)

    return experiment_name


def run_solver_with_mlflow(cfg: DictConfig) -> str:
    """Run solver and log to MLflow. Returns run_id."""
    solver = create_solver(cfg)
    solver_name = cfg.solver.name

    # Build run name
    if solver_name.startswith("spectral"):
        run_name = f"{solver_name}_N{cfg.N + 1}"
    else:
        run_name = f"{solver_name}_N{cfg.N}"

    # Setup parent run tagging
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    run_tags = {"solver": solver_name}
    nested = False
    if parent_run_id:
        run_tags["mlflow.parentRunId"] = parent_run_id
        run_tags["parent_run_id"] = parent_run_id
        run_tags["sweep"] = "child"
        nested = True

    with mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested) as run:
        # Log params and config
        mlflow.log_params(solver.params.to_mlflow())
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

        # Solve
        log.info(f"Solving: {solver_name} N={cfg.N} Re={cfg.Re}")
        solver.solve()

        # Log metrics
        mlflow.log_metrics(solver.metrics.to_mlflow())
        if solver.time_series is not None:
            batch_metrics = solver.time_series.to_mlflow_batch()
            if batch_metrics:
                MlflowClient().log_batch(run_id=run.info.run_id, metrics=batch_metrics)

        # Log solution VTK
        with tempfile.TemporaryDirectory() as tmpdir:
            vtk_path = Path(tmpdir) / "solution.vts"
            solver.to_vtk().save(str(vtk_path))
            mlflow.log_artifact(str(vtk_path))

        log.info(
            f"Done: {solver.metrics.iterations} iter, "
            f"converged={solver.metrics.converged}, "
            f"time={solver.metrics.wall_time_seconds:.2f}s"
        )

        return run.info.run_id


def generate_plots(cfg: DictConfig, run_id: str):
    """Generate plots for a completed run."""
    from shared.plotting.ldc import generate_plots_for_run

    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")

    generate_plots_for_run(
        run_id=run_id,
        tracking_uri=tracking_uri,
        output_dir=output_dir,
        solver_name=cfg.solver.name,
        N=cfg.N,
        Re=cfg.Re,
        parent_run_id=parent_run_id,
        upload_to_mlflow=True,
    )


def find_existing_run(cfg: DictConfig) -> str:
    """Find existing MLflow run matching config parameters."""
    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get experiment
    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    # Search for matching run
    filter_string = (
        f"params.Re = '{cfg.Re}' AND "
        f"params.nx = '{cfg.N}' AND "
        f"attributes.status = 'FINISHED'"
    )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No matching run found for N={cfg.N}, Re={cfg.Re}")

    run = runs[0]
    log.info(f"Found existing run: {run.info.run_name} ({run.info.run_id[:8]})")
    return run.info.run_id


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    log.info(f"Solver: {cfg.solver.name}, N={cfg.N}, Re={cfg.Re}")

    experiment_name = setup_mlflow(cfg)
    log.info(f"MLflow experiment: {experiment_name}")

    plot_only = cfg.get("plot_only", False)

    if plot_only:
        # Find existing run and regenerate plots
        run_id = find_existing_run(cfg)
        generate_plots(cfg, run_id)
    else:
        # Run solver then generate plots
        run_id = run_solver_with_mlflow(cfg)
        generate_plots(cfg, run_id)


if __name__ == "__main__":
    main()
