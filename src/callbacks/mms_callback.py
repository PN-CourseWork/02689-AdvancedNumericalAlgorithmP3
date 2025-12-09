"""Hydra callback for MMS test post-multirun plotting."""

import json
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import mlflow
import matplotlib.pyplot as plt
from hydra.experimental.callback import Callback
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


class MMSPlotCallback(Callback):
    """Generate convergence plots after multirun sweep completes."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Collect all results and generate plots per Reynolds number."""
        # Find the most recent multirun directory
        multirun_base = Path("multirun")
        if not multirun_base.exists():
            print("No multirun directory found")
            return

        # Find the latest date/time directory by modification time
        date_dirs = list(multirun_base.glob("*/*"))
        if not date_dirs:
            print("No multirun outputs found")
            return
        multirun_dir = max(date_dirs, key=lambda p: p.stat().st_mtime)
        print(f"Collecting results from {multirun_dir}")

        # Collect all mms_result.json files from job subdirectories
        results = []
        for job_dir in sorted(multirun_dir.glob("*")):
            if job_dir.is_dir():
                result_file = job_dir / "mms_result.json"
                if result_file.exists():
                    with open(result_file) as f:
                        results.append(json.load(f))

        if not results:
            print(f"No MMS results found in {multirun_dir}")
            return

        # Group by Reynolds number
        by_re = defaultdict(list)
        for r in results:
            by_re[r['Re']].append(r)

        # Create figures directory
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)

        # Generate one plot per Re and collect paths for MLflow
        plot_files = []
        for Re, data in by_re.items():
            plot_file = self._plot_convergence(Re, data, figures_dir)
            plot_files.append(plot_file)

        print(f"\nGenerated {len(by_re)} convergence plot(s) in {figures_dir}")

        # Log plots to MLflow parent run (if exists) and end it
        try:
            mlflow.set_experiment("MMS-Validation")
            experiment = mlflow.get_experiment_by_name("MMS-Validation")
            client = mlflow.tracking.MlflowClient()

            # Find the most recent RUNNING parent run (not yet terminated)
            parent_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.is_parent = 'true' AND attributes.status = 'RUNNING'",
                max_results=1,
                order_by=["start_time DESC"],
            )

            if parent_runs:
                parent_run_id = parent_runs[0].info.run_id
                print(f"Logging convergence plots to parent run {parent_run_id[:8]}...")
                for plot_file in plot_files:
                    client.log_artifact(parent_run_id, str(plot_file))
                print(f"Logged {len(plot_files)} plot(s) to MLflow parent run")

                # End the parent run to mark it as finished
                client.set_terminated(parent_run_id)
                print(f"Parent run {parent_run_id[:8]} marked as finished")
            else:
                print("Warning: No parent run found, skipping MLflow artifact logging")
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")

    def _plot_convergence(self, Re: float, data: list, output_dir: Path) -> Path:
        """Create convergence plot for a single Reynolds number."""
        # Sort by N
        data = sorted(data, key=lambda x: x['N'])
        N_vals = np.array([d['N'] for d in data])
        u_errors = np.array([d['u_error'] for d in data])
        v_errors = np.array([d['v_error'] for d in data])

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.semilogy(N_vals, u_errors, 'o-', label='u error', markersize=8)
        ax.semilogy(N_vals, v_errors, 's-', label='v error', markersize=8)

        ax.set_xlabel('N (polynomial degree)', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title(f'MMS Spectral Convergence (Re={Re})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Set x-axis to show integer N values
        ax.set_xticks(N_vals)

        plt.tight_layout()
        output_file = output_dir / f"mms_convergence_Re{int(Re)}.pdf"
        plt.savefig(output_file)
        plt.close()
        print(f"  Saved: {output_file}")
        return output_file
