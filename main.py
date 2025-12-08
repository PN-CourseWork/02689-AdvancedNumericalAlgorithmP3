"""
LDC Solver - Unified entry point for solving and plotting.

Usage:
    uv run python main.py -m +experiment/validation/ghia=fv
    uv run python main.py -m +experiment/validation/ghia=fv plot_only=true
    uv run python main.py solver=fv N=32 Re=100
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
from omegaconf import DictConfig, OmegaConf

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

log = logging.getLogger(__name__)


def get_experiment_name(cfg: DictConfig) -> str:
    """Build full experiment name with optional prefix."""
    name = cfg.experiment_name
    prefix = cfg.mlflow.get("project_prefix", "")
    if prefix and not name.startswith("/"):
        return f"{prefix}/{name}"
    return name


def setup_mlflow(cfg: DictConfig) -> str:
    """Setup MLflow tracking and return experiment name."""
    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    if str(cfg.mlflow.get("mode", "")).lower() in ("files", "local"):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ["MLFLOW_TRACKING_URI"] = str(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = get_experiment_name(cfg)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        experiment_name = f"{experiment_name}-restored"
        log.warning(f"MLflow set_experiment failed ({exc}); using '{experiment_name}'")
        mlflow.set_experiment(experiment_name)

    return experiment_name


def find_existing_run(cfg: DictConfig) -> str:
    """Find existing MLflow run matching config parameters."""
    experiment = mlflow.get_experiment_by_name(get_experiment_name(cfg))
    if not experiment:
        raise ValueError(f"Experiment not found: {cfg.experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.Re = '{cfg.Re}' AND params.nx = '{cfg.N}' AND attributes.status = 'FINISHED'",
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No matching run found for N={cfg.N}, Re={cfg.Re}")

    run_id = runs.iloc[0]["run_id"]
    log.info(f"Found existing run: {run_id[:8]}")
    return run_id


def run_solver(cfg: DictConfig) -> str:
    """Run solver and log to MLflow. Returns run_id."""
    solver = instantiate(cfg.solver, _convert_="partial")
    solver_name = cfg.solver.name

    # Run name: spectral uses N+1 (Chebyshev points)
    N_display = cfg.N + 1 if solver_name.startswith("spectral") else cfg.N
    run_name = f"{solver_name}_N{N_display}"

    # Parent run tagging for sweeps
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    tags = {"solver": solver_name}
    if parent_run_id:
        tags.update({"mlflow.parentRunId": parent_run_id, "parent_run_id": parent_run_id, "sweep": "child"})

    with mlflow.start_run(run_name=run_name, tags=tags, nested=bool(parent_run_id)) as run:
        mlflow.log_params(solver.params.to_mlflow())
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

        log.info(f"Solving: {solver_name} N={cfg.N} Re={cfg.Re}")
        solver.solve()

        # Compute validation errors against reference FV solution
        reference_dir = cfg.get("validation", {}).get("reference_dir", "data/validation/fv")
        validation_errors = solver.compute_validation_errors(reference_dir=reference_dir)
        if validation_errors:
            mlflow.log_metrics(validation_errors)

        mlflow.log_metrics(solver.metrics.to_mlflow())
        if solver.time_series:
            batch = solver.time_series.to_mlflow_batch()
            if batch:
                mlflow.tracking.MlflowClient().log_batch(run.info.run_id, metrics=batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            vtk_path = Path(tmpdir) / "solution.vts"
            solver.to_vtk().save(str(vtk_path))
            mlflow.log_artifact(str(vtk_path))

        log.info(f"Done: {solver.metrics.iterations} iter, converged={solver.metrics.converged}, time={solver.metrics.wall_time_seconds:.2f}s")
        return run.info.run_id


def generate_plots(cfg: DictConfig, run_id: str):
    """Generate plots for a completed run."""
    from shared.plotting.ldc import generate_plots_for_run

    generate_plots_for_run(
        run_id=run_id,
        tracking_uri=cfg.mlflow.get("tracking_uri", "./mlruns"),
        output_dir=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
        solver_name=cfg.solver.name,
        N=cfg.N,
        Re=cfg.Re,
        parent_run_id=os.environ.get("MLFLOW_PARENT_RUN_ID"),
        upload_to_mlflow=True,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    log.info(f"Solver: {cfg.solver.name}, N={cfg.N}, Re={cfg.Re}")
    log.info(f"MLflow experiment: {setup_mlflow(cfg)}")

    run_id = find_existing_run(cfg) if cfg.get("plot_only") else run_solver(cfg)
    generate_plots(cfg, run_id)


if __name__ == "__main__":
    main()
