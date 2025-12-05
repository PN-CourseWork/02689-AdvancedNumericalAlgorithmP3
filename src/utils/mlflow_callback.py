"""Hydra callback for MLflow parent run management during sweeps."""

import logging
import os
from typing import Dict, Optional

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class MLflowSweepCallback(Callback):
    """Creates or reuses parent MLflow runs for Hydra multiruns.

    Supports grouping by Reynolds number (or other parameters):
    - If sweep_name contains ${Re}, separate parent runs are created per Re value
    - Child runs are nested under their respective parent
    - Makes it easy to compare solvers at the same Re

    Example sweep_name patterns:
    - "my-sweep"           -> Single parent for all runs
    - "my-sweep-Re${Re}"   -> Separate parent per Reynolds number
    """

    def __init__(self) -> None:
        self._parent_runs: Dict[str, str] = {}  # sweep_name -> run_id
        self._active_parent_runs: Dict[str, object] = {}  # sweep_name -> run object
        self._current_parent_id: Optional[str] = None
        self._tracking_uri: Optional[str] = None
        self._full_experiment_name: Optional[str] = None
        self._base_sweep_name: Optional[str] = None
        self._sweep_dir: Optional[str] = None  # Store sweep dir while HydraConfig available

    def _find_existing_parent(self, experiment_name: str, sweep_name: str) -> Optional[str]:
        """Find an existing parent run with the same sweep_name."""
        import mlflow

        try:
            runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.sweep = 'parent' AND tags.`mlflow.runName` = '{sweep_name}'",
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs.empty:
                return None

            return runs.iloc[0]["run_id"]

        except Exception as e:
            log.warning(f"Error searching for parent run: {e}")

        return None

    def _get_or_create_parent(self, sweep_name: str, config: DictConfig) -> str:
        """Get existing parent run or create a new one for this sweep_name."""
        import mlflow

        # Check our cache first
        if sweep_name in self._parent_runs:
            return self._parent_runs[sweep_name]

        # Check for existing parent in MLflow
        existing_id = self._find_existing_parent(self._full_experiment_name, sweep_name)
        if existing_id:
            self._parent_runs[sweep_name] = existing_id
            log.info(f"Reusing existing parent run '{sweep_name}': {existing_id}")
            return existing_id

        # Create new parent run
        parent_run = mlflow.start_run(run_name=sweep_name)
        parent_id = parent_run.info.run_id
        self._parent_runs[sweep_name] = parent_id
        self._active_parent_runs[sweep_name] = parent_run

        # Log config and tags to parent
        mlflow.log_dict(OmegaConf.to_container(config), "sweep_config.yaml")
        mlflow.set_tag("sweep", "parent")

        # Extract Re from sweep_name if present
        if "Re" in sweep_name:
            # Try to extract Re value from sweep_name like "sweep-Re100"
            import re
            match = re.search(r'Re(\d+)', sweep_name)
            if match:
                mlflow.set_tag("Re", match.group(1))

        # Log HPC job info if available
        job_id = os.environ.get("LSB_JOBID")
        if job_id:
            mlflow.set_tag("lsf.job_id", job_id)
            mlflow.set_tag("lsf.job_name", os.environ.get("LSB_JOBNAME", ""))

        # End the run context (we'll reference it by ID)
        mlflow.end_run()

        log.info(f"Created parent run '{sweep_name}': {parent_id}")
        return parent_id

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        """Setup MLflow tracking before sweep starts."""
        import mlflow
        from dotenv import load_dotenv

        load_dotenv()

        # Setup MLflow tracking
        self._tracking_uri = config.mlflow.get("tracking_uri", "./mlruns")
        mlflow.set_tracking_uri(self._tracking_uri)

        # Build experiment name
        experiment_name = config.experiment_name
        project_prefix = config.mlflow.get("project_prefix", "")
        if project_prefix and not experiment_name.startswith("/"):
            self._full_experiment_name = f"{project_prefix}/{experiment_name}"
        else:
            self._full_experiment_name = experiment_name

        mlflow.set_experiment(self._full_experiment_name)

        # Store base sweep name (may contain ${Re} placeholder)
        self._base_sweep_name = config.get("sweep_name", "sweep")

        # Store sweep directory while HydraConfig is available
        try:
            import hydra.core.hydra_config
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            self._sweep_dir = hydra_cfg.sweep.dir
        except Exception:
            self._sweep_dir = None

        log.info(f"MLflow sweep callback initialized for experiment: {self._full_experiment_name}")

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        """Set parent run ID for each job based on its Re value."""
        import mlflow

        # Ensure tracking is set up
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        if self._full_experiment_name:
            mlflow.set_experiment(self._full_experiment_name)

        # Resolve sweep_name with current job's config (e.g., Re value)
        sweep_name = self._base_sweep_name
        if "${Re}" in sweep_name or "{Re}" in sweep_name:
            re_value = int(config.get("Re", 100))
            sweep_name = sweep_name.replace("${Re}", str(re_value)).replace("{Re}", str(re_value))

        # Get or create parent for this sweep_name
        parent_id = self._get_or_create_parent(sweep_name, config)
        self._current_parent_id = parent_id

        # Set env var so child run can find parent
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_id

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        """Clean up after sweep completes and generate comparison plots."""
        # Clean up env var
        os.environ.pop("MLFLOW_PARENT_RUN_ID", None)

        # Log summary
        log.info(f"Sweep completed. Created {len(self._active_parent_runs)} parent run(s).")
        for name, run_id in self._parent_runs.items():
            log.info(f"  - {name}: {run_id}")

        # Generate comparison plots for all parent runs
        if self._parent_runs and config.get("generate_plots", True):
            try:
                from plotting import generate_comparison_plots_for_sweep
                from pathlib import Path

                # Use stored sweep directory (HydraConfig may not be available here)
                if self._sweep_dir:
                    output_dir = Path(self._sweep_dir) / "comparison_plots"
                else:
                    output_dir = Path("outputs") / "comparison_plots"

                parent_run_ids = list(self._parent_runs.values())
                log.info(f"Generating comparison plots for {len(parent_run_ids)} parent run(s)...")

                generate_comparison_plots_for_sweep(
                    parent_run_ids=parent_run_ids,
                    tracking_uri=self._tracking_uri,
                    output_dir=output_dir,
                    upload_to_mlflow=True,
                )
            except Exception as e:
                log.warning(f"Failed to generate comparison plots: {e}")
