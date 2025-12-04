"""Hydra callback for MLflow parent run management during sweeps."""

import logging
import os
from datetime import datetime, timedelta

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class MLflowSweepCallback(Callback):
    """Creates or reuses a parent MLflow run for Hydra multiruns.

    Child runs will be nested under the parent, making it easy to:
    - View all sweep runs together
    - Log sweep-level config to parent
    - Group multiple sweeps under the same parent (shared sweep_name)

    If a parent run with the same sweep_name exists and was created within
    the last hour, it will be reused. Otherwise, a new parent is created.
    """

    def __init__(self) -> None:
        self._parent_run = None
        self._parent_run_id = None
        self._owns_parent = False  # Whether we created the parent (vs reusing)

    def _find_recent_parent(self, experiment_name: str, sweep_name: str, max_age_hours: float = 1.0):
        """Find a recent parent run with the same sweep_name."""
        import mlflow

        try:
            # Search for parent runs with matching name
            runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.sweep = 'parent' AND tags.`mlflow.runName` = '{sweep_name}'",
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs.empty:
                return None

            # Check if recent enough
            run = runs.iloc[0]
            start_time = run["start_time"]

            # Handle timezone-aware timestamps
            if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                now = datetime.now(start_time.tzinfo)
            else:
                now = datetime.now()

            age = now - start_time
            if age < timedelta(hours=max_age_hours):
                return run["run_id"]

        except Exception as e:
            log.warning(f"Error searching for parent run: {e}")

        return None

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        """Create or reuse parent run before sweep starts."""
        import mlflow
        from dotenv import load_dotenv

        load_dotenv()

        # Setup MLflow tracking
        tracking_uri = config.mlflow.get("tracking_uri", "./mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        # Build experiment name
        experiment_name = config.experiment_name
        project_prefix = config.mlflow.get("project_prefix", "")
        if project_prefix and not experiment_name.startswith("/"):
            full_experiment_name = f"{project_prefix}/{experiment_name}"
        else:
            full_experiment_name = experiment_name

        mlflow.set_experiment(full_experiment_name)

        sweep_name = config.get("sweep_name", "sweep")

        # Check for existing recent parent run with same name
        existing_parent_id = self._find_recent_parent(full_experiment_name, sweep_name)

        if existing_parent_id:
            # Reuse existing parent
            self._parent_run_id = existing_parent_id
            self._owns_parent = False
            log.info(f"Reusing existing parent run: {self._parent_run_id}")
        else:
            # Create new parent run
            self._parent_run = mlflow.start_run(run_name=sweep_name)
            self._parent_run_id = self._parent_run.info.run_id
            self._owns_parent = True

            # Log sweep config to parent
            mlflow.log_dict(OmegaConf.to_container(config), "sweep_config.yaml")
            mlflow.set_tag("sweep", "parent")

            # Log HPC job info if available
            job_id = os.environ.get("LSB_JOBID")
            if job_id:
                mlflow.set_tag("lsf.job_id", job_id)
                mlflow.set_tag("lsf.job_name", os.environ.get("LSB_JOBNAME", ""))

            log.info(f"Created parent run: {self._parent_run_id}")

        # Set env var so child processes can find parent
        os.environ["MLFLOW_PARENT_RUN_ID"] = self._parent_run_id

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        """End parent run after sweep completes (only if we created it)."""
        import mlflow

        if self._owns_parent and self._parent_run is not None:
            mlflow.end_run()
            log.info(f"Ended parent run: {self._parent_run_id}")

        # Clean up env var
        os.environ.pop("MLFLOW_PARENT_RUN_ID", None)
