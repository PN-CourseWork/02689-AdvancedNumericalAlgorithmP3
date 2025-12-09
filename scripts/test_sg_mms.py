"""Method of Manufactured Solutions (MMS) Test for SGSolver.

Manufactured solution (satisfies div(u) = 0):
  u = sin(pi*x) * cos(pi*y)
  v = -cos(pi*x) * sin(pi*y)
  p = sin(pi*x) * sin(pi*y)

"""

import json
import os
from pathlib import Path
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import sys
sys.path.insert(0, 'src')

from solvers.spectral.sg import SGSolver

# Parent run ID stored globally for child runs to access
_PARENT_RUN_ID = None


def manufactured_solution(x, y):
    """Return exact manufactured solution u, v at coordinates (x, y)."""
    u = np.sin(np.pi * x) * np.cos(np.pi * y)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y)
    return u, v


def mms_forcing_u(x, y, Re=100):
    """Compute MMS forcing term for u-momentum equation."""
    u = np.sin(np.pi * x) * np.cos(np.pi * y)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y)

    du_dx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
    du_dy = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)

    lap_u = -2 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
    dp_dx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)

    conv_u = u * du_dx + v * du_dy
    return conv_u + dp_dx - (1.0/Re) * lap_u


def mms_forcing_v(x, y, Re=100):
    """Compute MMS forcing term for v-momentum equation."""
    u = np.sin(np.pi * x) * np.cos(np.pi * y)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y)

    dv_dx = np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
    dv_dy = -np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)

    lap_v = 2 * np.pi**2 * np.cos(np.pi * x) * np.sin(np.pi * y)
    dp_dy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

    conv_v = u * dv_dx + v * dv_dy
    return conv_v + dp_dy - (1.0/Re) * lap_v


def plot_mms_fields(solver, u_exact, v_exact, u_diff, v_diff, N, Re, output_dir):
    """Create solution field plots for MMS test."""
    X, Y = solver.x_full, solver.y_full
    u_computed = solver.arrays.u.reshape(solver.shape_full)
    v_computed = solver.arrays.v.reshape(solver.shape_full)

    # Create figure with 3 rows (u, v, error) x 3 cols (computed, exact, diff)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # U velocity
    vmax_u = max(abs(u_computed.max()), abs(u_computed.min()), abs(u_exact.max()), abs(u_exact.min()))
    im = axes[0, 0].contourf(X, Y, u_computed, levels=20, cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u)
    axes[0, 0].set_title('u (computed)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].contourf(X, Y, u_exact, levels=20, cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u)
    axes[0, 1].set_title('u (exact)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].contourf(X, Y, u_diff, levels=20, cmap='RdBu_r')
    axes[0, 2].set_title(f'u error (max={np.max(np.abs(u_diff)):.2e})')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 2])

    # V velocity
    vmax_v = max(abs(v_computed.max()), abs(v_computed.min()), abs(v_exact.max()), abs(v_exact.min()))
    im = axes[1, 0].contourf(X, Y, v_computed, levels=20, cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v)
    axes[1, 0].set_title('v (computed)')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].contourf(X, Y, v_exact, levels=20, cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v)
    axes[1, 1].set_title('v (exact)')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].contourf(X, Y, v_diff, levels=20, cmap='RdBu_r')
    axes[1, 2].set_title(f'v error (max={np.max(np.abs(v_diff)):.2e})')
    axes[1, 2].set_aspect('equal')
    plt.colorbar(im, ax=axes[1, 2])

    fig.suptitle(f'MMS Test: N={N}, Re={Re}', fontsize=14)
    plt.tight_layout()

    output_file = output_dir / f"mms_fields_N{N}_Re{int(Re)}.pdf"
    plt.savefig(output_file)
    plt.close()
    return output_file


def run_mms_test(cfg: DictConfig, output_dir: Path = None) -> dict:
    """Run MMS test for a single N value."""
    N = cfg.N
    Re = cfg.Re

    def forcing_u(x, y):
        return mms_forcing_u(x, y, Re)

    def forcing_v(x, y):
        return mms_forcing_v(x, y, Re)

    solver = SGSolver(
        nx=N, ny=N,
        Re=Re,
        tolerance=cfg.tolerance,
        max_iterations=cfg.max_iterations,
        CFL=cfg.CFL,
        corner_treatment=cfg.corner_treatment,
        corner_smoothing=cfg.corner_smoothing,
        forcing_u=forcing_u,
        forcing_v=forcing_v,
        bc_func=manufactured_solution,
    )

    solver.solve()

    # Get exact solution on the grid
    u_exact, v_exact = manufactured_solution(solver.x_full, solver.y_full)

    # Compute L2 errors using quadrature weights
    W = solver.W_2d
    u_computed = solver.arrays.u.reshape(solver.shape_full)
    v_computed = solver.arrays.v.reshape(solver.shape_full)

    u_diff = u_computed - u_exact
    v_diff = v_computed - v_exact

    # Compute absolute L2 errors (not relative)
    u_L2 = np.sqrt(np.sum(W * u_diff**2))
    v_L2 = np.sqrt(np.sum(W * v_diff**2))

    # Create solution field plots if output directory specified
    field_plot = None
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        field_plot = plot_mms_fields(solver, u_exact, v_exact, u_diff, v_diff, N, Re, output_dir)

    return {
        'N': N,
        'u_error': u_L2,
        'v_error': v_L2,
        'converged': solver.metrics.converged,
        'iterations': solver.metrics.iterations,
        'field_plot': field_plot,
    }


def get_or_create_parent_run(experiment_name: str, parent_run_name: str):

    global _PARENT_RUN_ID

    # Check if we already have a parent run ID (within same process)
    if _PARENT_RUN_ID is not None:
        return _PARENT_RUN_ID

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    client = mlflow.tracking.MlflowClient()

    # Search for existing RUNNING parent run (status = RUNNING means not yet finished)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}' AND tags.is_parent = 'true' AND attributes.status = 'RUNNING'",
        max_results=1,
    )

    if runs:
        _PARENT_RUN_ID = runs[0].info.run_id
    else:
        # Create new parent run
        parent_run = client.create_run(
            experiment_id=experiment.experiment_id,
            run_name=parent_run_name,
            tags={"is_parent": "true"},
        )
        _PARENT_RUN_ID = parent_run.info.run_id

    return _PARENT_RUN_ID


@hydra.main(version_base=None, config_path="../conf/experiment/validation/mms", config_name="spectral")
def main(cfg: DictConfig):
    global _PARENT_RUN_ID

    # Setup MLflow tracking
    experiment_name = "MMS-Validation"
    mlflow.set_experiment(experiment_name)

    parent_run_name = os.environ.get("MMS_PARENT_RUN", "MMS_Sweep")

    # Get or create parent run 
    parent_run_id = get_or_create_parent_run(experiment_name, parent_run_name)

    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    child_run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name=f"N{cfg.N}_Re{int(cfg.Re)}",
        tags={"mlflow.parentRunId": parent_run_id},
    )
    child_run_id = child_run.info.run_id

    # Use the child run for logging
    with mlflow.start_run(run_id=child_run_id, log_system_metrics=True):
        # Log parameters
        mlflow.log_params({
            'N': cfg.N,
            'Re': cfg.Re,
            'tolerance': cfg.tolerance,
            'max_iterations': cfg.max_iterations,
            'CFL': cfg.CFL,
            'corner_treatment': cfg.corner_treatment,
            'corner_smoothing': cfg.corner_smoothing,
        })

        job_id = os.environ.get("LSB_JOBID")
        if job_id:
            mlflow.set_tag("lsf.job_id", job_id)

        figures_dir = Path("figures")
        result = run_mms_test(cfg, output_dir=figures_dir)

        converged_str = "Yes" if result['converged'] else "No"
        print(f"\n{'='*70}")
        print(f"MMS Test Result: N={result['N']}, Re={cfg.Re}")
        print(f"{'='*70}")
        print(f"  u L2 error:  {result['u_error']:.6e}")
        print(f"  v L2 error:  {result['v_error']:.6e}")
        print(f"  Converged:   {converged_str}")
        print(f"  Iterations:  {result['iterations']}")

        # Log metrics to MLflow
        mlflow.log_metrics({
            'u_error': float(result['u_error']),
            'v_error': float(result['v_error']),
            'iterations': int(result['iterations']),
            'converged': int(result['converged']),
        })

        # Log field plot as artifact
        if result['field_plot']:
            mlflow.log_artifact(str(result['field_plot']))
            print(f"  Field plot:  {result['field_plot']}")

        # Save result to JSON for post-multirun a
        result_json = {
            'N': int(result['N']),
            'Re': float(cfg.Re),
            'u_error': float(result['u_error']),
            'v_error': float(result['v_error']),
            'converged': bool(result['converged']),
            'iterations': int(result['iterations']),
        }
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        with open(f'{hydra_output_dir}/mms_result.json', 'w') as f:
            json.dump(result_json, f)

    return result['u_error']


if __name__ == "__main__":
    main()
