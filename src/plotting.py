"""
Solution Plotter - Generates plots for MLflow runs.

Can be used standalone via Hydra or called directly from run_solver.py.

For individual runs: generates fields, streamlines, vorticity, centerlines,
                     ghia_validation, and convergence plots.

For sweeps: additionally generates a Ghia comparison plot uploaded to parent run.

Usage (standalone):
    # Plot single run
    uv run python src/plotting.py solver=spectral N=31 Re=100

    # Plot sweep (comparison plot goes to parent)
    uv run python src/plotting.py -m solver=spectral N=15,31,63 Re=100

Usage (from run_solver.py):
    from plotting import generate_plots_for_run
    generate_plots_for_run(run_id, tracking_uri, output_dir, parent_run_id)
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import hydra
import mlflow
import zarr
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# =============================================================================
# MLflow Run Discovery
# =============================================================================


def find_matching_run(cfg: DictConfig, tracking_uri: str) -> tuple[str, Optional[str]]:
    """Find MLflow run matching the config parameters.

    Returns
    -------
    tuple[str, Optional[str]]
        (run_id, parent_run_id) - parent_run_id is None if not a sweep child
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment_name = cfg.experiment_name
    project_prefix = cfg.mlflow.get("project_prefix", "")
    if project_prefix and not experiment_name.startswith("/"):
        experiment_name = f"{project_prefix}/{experiment_name}"

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    # Build filter string for matching runs
    solver_name = cfg.solver.name
    N = cfg.N
    Re = cfg.Re

    filter_parts = [
        f"params.Re = '{Re}'",
        f"params.nx = '{N}'",
        f"params.ny = '{N}'",
        "attributes.status = 'FINISHED'",
    ]

    # Add solver-specific filter
    if solver_name == "spectral":
        filter_parts.append(f"params.basis_type = '{cfg.solver.basis_type}'")
    elif solver_name == "fv":
        filter_parts.append(f"params.convection_scheme = '{cfg.solver.convection_scheme}'")

    filter_string = " AND ".join(filter_parts)

    log.info(f"Searching in experiment: {experiment_name}")
    log.info(f"Filter: solver={solver_name}, N={N}, Re={Re}")

    # Search for runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )

    if not runs:
        raise ValueError(
            f"No matching runs found for solver={solver_name}, N={N}, Re={Re}\n"
            f"Filter used: {filter_string}"
        )

    # Return most recent matching run
    run = runs[0]
    parent_run_id = run.data.tags.get("parent_run_id")

    log.info(f"Found run: {run.info.run_name} (id: {run.info.run_id[:8]}...)")
    if parent_run_id:
        log.info(f"  Parent run: {parent_run_id[:8]}...")

    return run.info.run_id, parent_run_id


def find_sibling_runs(parent_run_id: str, tracking_uri: str) -> list[dict]:
    """Find all child runs of a parent (siblings in a sweep).

    Returns list of dicts with run info for comparison plotting.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get parent run to find experiment
    parent_run = client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id

    # Find all children of this parent (include RUNNING for parallel sweeps)
    filter_string = f"tags.parent_run_id = '{parent_run_id}'"

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["params.nx ASC"],  # Sort by N for nice legend order
        max_results=50,
    )

    siblings = []
    for run in runs:
        run_name = run.info.run_name or ""

        # Extract solver name from run_name (format: {solver}_N{n} or {solver}_N{n}_Re{re})
        # Examples: "fv_N32", "spectral_N33", "spectral_fsg_N16"
        if "_N" in run_name:
            solver_name = run_name.rsplit("_N", 1)[0]  # rsplit to handle underscores in solver name
        else:
            solver_name = "unknown"

        siblings.append({
            "run_id": run.info.run_id,
            "run_name": run_name,
            "N": int(run.data.params.get("nx", 0)),
            "Re": float(run.data.params.get("Re", 0)),
            "solver": solver_name,
            "status": run.info.status,
        })

    finished = sum(1 for s in siblings if s["status"] == "FINISHED")
    log.info(f"Found {len(siblings)} sibling runs in sweep ({finished} finished)")
    return siblings


# =============================================================================
# Data Loading from MLflow
# =============================================================================


def download_mlflow_artifacts(run_id: str, tracking_uri: str) -> Path:
    """Download solution artifacts from MLflow run to temp directory."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    log.info(f"Downloading artifacts from: {run.info.run_name}")

    tmpdir = tempfile.mkdtemp(prefix="ldc_plot_")
    artifact_path = client.download_artifacts(run_id, "", tmpdir)

    return Path(artifact_path)


def load_fields_from_zarr(artifact_dir: Path) -> dict:
    """Load solution fields from zarr artifacts."""
    fields_dir = artifact_dir / "fields"
    if not fields_dir.exists():
        raise FileNotFoundError(f"Fields directory not found: {fields_dir}")

    fields = {}
    for name in ["x", "y", "u", "v", "p"]:
        zarr_path = fields_dir / f"{name}.zarr"
        if zarr_path.exists():
            fields[name] = zarr.load(zarr_path)
        else:
            raise FileNotFoundError(f"Field not found: {zarr_path}")

    return fields


def restructure_fields(fields: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert potentially flattened fields to structured 2D arrays.

    Returns (x_unique, y_unique, U_2d, V_2d, P_2d) where U, V, P are 2D arrays
    with shape (ny, nx) suitable for RectBivariateSpline.
    """
    x, y = fields["x"], fields["y"]
    u, v, p = fields["u"], fields["v"], fields["p"]

    # Get unique coordinates
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))
    nx, ny = len(x_unique), len(y_unique)

    # If already 2D, just return
    if u.ndim == 2:
        return x_unique, y_unique, u, v, p

    # Reshape from flattened to 2D
    # Create mapping from coordinates to indices
    x_to_idx = {val: i for i, val in enumerate(x_unique)}
    y_to_idx = {val: i for i, val in enumerate(y_unique)}

    U_2d = np.zeros((ny, nx))
    V_2d = np.zeros((ny, nx))
    P_2d = np.zeros((ny, nx))

    for k in range(len(x)):
        i = x_to_idx[x[k]]
        j = y_to_idx[y[k]]
        U_2d[j, i] = u[k]
        V_2d[j, i] = v[k]
        P_2d[j, i] = p[k]

    return x_unique, y_unique, U_2d, V_2d, P_2d


def load_timeseries_from_mlflow(run_id: str, tracking_uri: str) -> pd.DataFrame:
    """Load timeseries metrics from MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get metric history
    metrics_to_fetch = ["residual", "energy", "enstrophy", "palinstrophy"]

    data = {}
    for metric_name in metrics_to_fetch:
        try:
            history = client.get_metric_history(run_id, metric_name)
            if history:
                data[metric_name] = [m.value for m in sorted(history, key=lambda x: x.step)]
        except Exception:
            pass  # Metric might not exist

    if not data:
        return pd.DataFrame()

    # Create DataFrame with iteration index
    max_len = max(len(v) for v in data.values())
    df = pd.DataFrame({k: v + [None] * (max_len - len(v)) for k, v in data.items()})
    df["iteration"] = range(len(df))

    return df


def fields_to_dataframe(fields: dict) -> pd.DataFrame:
    """Convert fields dict to DataFrame format for plotting.

    Handles two storage formats:
    1. Flattened: x, y, u, v, p all 1D with same length (already meshgrid coords)
    2. Structured: x, y are 1D coordinate arrays, u, v, p are 2D fields
    """
    x, y = fields["x"], fields["y"]
    u, v, p = fields["u"], fields["v"], fields["p"]

    # Check if already flattened (all same length, 1D)
    if x.ndim == 1 and len(x) == len(u.ravel()):
        # Already flattened meshgrid coordinates
        return pd.DataFrame({
            "x": x,
            "y": y,
            "u": u.ravel(),
            "v": v.ravel(),
            "p": p.ravel(),
        })

    # Need to create meshgrid
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y

    return pd.DataFrame({
        "x": X.ravel(),
        "y": Y.ravel(),
        "u": u.ravel(),
        "v": v.ravel(),
        "p": p.ravel(),
    })


# =============================================================================
# Individual Run Plotting Functions
# =============================================================================


def plot_fields(fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path) -> Path:
    """Generate field contour plots (p, u, v)."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline

    x_unique = np.sort(fields_df["x"].unique())
    y_unique = np.sort(fields_df["y"].unique())
    nx, ny = len(x_unique), len(y_unique)

    sorted_df = fields_df.sort_values(["y", "x"])
    P = sorted_df["p"].values.reshape(ny, nx)
    U = sorted_df["u"].values.reshape(ny, nx)
    V = sorted_df["v"].values.reshape(ny, nx)

    n_fine = 200
    x_fine = np.linspace(x_unique[0], x_unique[-1], n_fine)
    y_fine = np.linspace(y_unique[0], y_unique[-1], n_fine)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    P_interp = RectBivariateSpline(y_unique, x_unique, P)(y_fine, x_fine)
    U_interp = RectBivariateSpline(y_unique, x_unique, U)(y_fine, x_fine)
    V_interp = RectBivariateSpline(y_unique, x_unique, V)(y_fine, x_fine)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cf_p = axes[0].contourf(X_fine, Y_fine, P_interp, levels=25, cmap="coolwarm")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Pressure", fontweight="bold")
    axes[0].set_aspect("equal")
    plt.colorbar(cf_p, ax=axes[0], label="p")

    cf_u = axes[1].contourf(X_fine, Y_fine, U_interp, levels=25, cmap="RdBu_r")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("U velocity", fontweight="bold")
    axes[1].set_aspect("equal")
    plt.colorbar(cf_u, ax=axes[1], label="u")

    cf_v = axes[2].contourf(X_fine, Y_fine, V_interp, levels=25, cmap="RdBu_r")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("V velocity", fontweight="bold")
    axes[2].set_aspect("equal")
    plt.colorbar(cf_v, ax=axes[2], label="v")

    fig.suptitle(f"Solution Fields — {solver.upper()} N={N}, Re={Re:.0f}", fontweight="bold", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / "fields.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_streamlines(fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path) -> Path:
    """Generate streamline plot with velocity magnitude."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline

    x_unique = np.sort(fields_df["x"].unique())
    y_unique = np.sort(fields_df["y"].unique())
    nx, ny = len(x_unique), len(y_unique)

    sorted_df = fields_df.sort_values(["y", "x"])
    U = sorted_df["u"].values.reshape(ny, nx)
    V = sorted_df["v"].values.reshape(ny, nx)

    n_fine = 200
    x_fine = np.linspace(x_unique[0], x_unique[-1], n_fine)
    y_fine = np.linspace(y_unique[0], y_unique[-1], n_fine)

    U_interp = RectBivariateSpline(y_unique, x_unique, U)(y_fine, x_fine)
    V_interp = RectBivariateSpline(y_unique, x_unique, V)(y_fine, x_fine)
    vel_mag = np.sqrt(U_interp**2 + V_interp**2)

    fig, ax = plt.subplots(figsize=(8, 7))

    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    cf = ax.contourf(X_fine, Y_fine, vel_mag, levels=25, cmap="viridis")
    stream = ax.streamplot(
        x_fine, y_fine, U_interp, V_interp,
        color="white", linewidth=0.8, density=1.5,
        arrowsize=1.0, arrowstyle="->"
    )
    stream.lines.set_alpha(0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Streamlines — {solver.upper()} N={N}, Re={Re:.0f}", fontweight="bold")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="Velocity magnitude")
    plt.tight_layout()

    output_path = output_dir / "streamlines.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_vorticity(fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path) -> Path:
    """Generate vorticity contour plot."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline

    x_unique = np.sort(fields_df["x"].unique())
    y_unique = np.sort(fields_df["y"].unique())
    nx, ny = len(x_unique), len(y_unique)

    sorted_df = fields_df.sort_values(["y", "x"])
    U = sorted_df["u"].values.reshape(ny, nx)
    V = sorted_df["v"].values.reshape(ny, nx)

    n_fine = 200
    x_fine = np.linspace(x_unique[0], x_unique[-1], n_fine)
    y_fine = np.linspace(y_unique[0], y_unique[-1], n_fine)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    U_spline = RectBivariateSpline(y_unique, x_unique, U)
    V_spline = RectBivariateSpline(y_unique, x_unique, V)

    dvdx = V_spline(y_fine, x_fine, dx=1)
    dudy = U_spline(y_fine, x_fine, dy=1)
    vorticity = dvdx - dudy

    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = np.max(np.abs(vorticity))
    cf = ax.contourf(X_fine, Y_fine, vorticity, levels=25, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Vorticity — {solver.upper()} N={N}, Re={Re:.0f}", fontweight="bold")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label=r"$\omega$")
    plt.tight_layout()

    output_path = output_dir / "vorticity.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_centerlines(fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path) -> Path:
    """Plot velocity profiles along centerlines."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline

    x_unique = np.sort(fields_df["x"].unique())
    y_unique = np.sort(fields_df["y"].unique())
    nx, ny = len(x_unique), len(y_unique)

    sorted_df = fields_df.sort_values(["y", "x"])
    U = sorted_df["u"].values.reshape(ny, nx)
    V = sorted_df["v"].values.reshape(ny, nx)

    U_spline = RectBivariateSpline(y_unique, x_unique, U)
    V_spline = RectBivariateSpline(y_unique, x_unique, V)

    n_points = 200
    y_line = np.linspace(y_unique[0], y_unique[-1], n_points)
    x_line = np.linspace(x_unique[0], x_unique[-1], n_points)

    x_center = (x_unique[0] + x_unique[-1]) / 2
    y_center = (y_unique[0] + y_unique[-1]) / 2

    u_vertical = U_spline(y_line, x_center).ravel()
    v_horizontal = V_spline(y_center, x_line).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(u_vertical, y_line, "b-", linewidth=2)
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("y")
    axes[0].set_title("U velocity (vertical centerline)", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    axes[1].plot(x_line, v_horizontal, "r-", linewidth=2)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].set_title("V velocity (horizontal centerline)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(f"Centerline Profiles — {solver.upper()} N={N}, Re={Re:.0f}", fontweight="bold", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / "centerlines.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_convergence(timeseries_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path) -> Path:
    """Plot convergence history (residuals over iterations)."""
    import matplotlib.pyplot as plt

    if timeseries_df.empty:
        log.warning("No timeseries data available for convergence plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot available metrics
    colors = {"residual": "blue", "energy": "green", "enstrophy": "orange", "palinstrophy": "red"}

    for col in timeseries_df.columns:
        if col != "iteration" and col in colors:
            data = timeseries_df[col].dropna()
            if len(data) > 0:
                ax.semilogy(timeseries_df.loc[data.index, "iteration"], data,
                           label=col.capitalize(), color=colors[col], linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(f"Convergence History — {solver.upper()} N={N}, Re={Re:.0f}", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "convergence.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_ghia_validation(fields: dict, Re: float, solver: str, N: int, output_dir: Path) -> Path:
    """Plot validation against Ghia benchmark data."""
    import matplotlib.pyplot as plt
    from spectral import spectral_interpolate

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]
    if int(Re) not in AVAILABLE_RE:
        log.warning(f"Ghia data not available for Re={Re}. Available: {AVAILABLE_RE}")
        return None

    # Project root is 1 level up from src/
    project_root = Path(__file__).parent.parent
    ghia_dir = project_root / "data" / "validation" / "ghia"

    u_file = ghia_dir / f"ghia_Re{int(Re)}_u_centerline.csv"
    v_file = ghia_dir / f"ghia_Re{int(Re)}_v_centerline.csv"

    if not u_file.exists() or not v_file.exists():
        log.warning(f"Ghia data files not found in {ghia_dir}")
        return None

    ghia_u = pd.read_csv(u_file)
    ghia_v = pd.read_csv(v_file)

    # Restructure fields to 2D arrays
    x_unique, y_unique, U_2d, V_2d, _ = restructure_fields(fields)

    # Use spectral interpolation for centerline extraction
    n_points = 200
    y_line = np.linspace(y_unique.min(), y_unique.max(), n_points)
    x_line = np.linspace(x_unique.min(), x_unique.max(), n_points)

    # Find center indices
    x_center_idx = len(x_unique) // 2
    y_center_idx = len(y_unique) // 2

    # U along vertical centerline (x = center): interpolate in y
    u_at_center_x = U_2d[:, x_center_idx]
    u_sim = spectral_interpolate(y_unique, u_at_center_x, y_line, basis="legendre")

    # V along horizontal centerline (y = center): interpolate in x
    v_at_center_y = V_2d[y_center_idx, :]
    v_sim = spectral_interpolate(x_unique, v_at_center_y, x_line, basis="legendre")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(u_sim, y_line, "b-", linewidth=2, label=f"{solver.upper()} N={N}")
    axes[0].scatter(ghia_u["u"], ghia_u["y"], c="black", s=50, marker="x",
                    label="Ghia et al. (1982)", zorder=10)
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("y")
    axes[0].set_title("U velocity (vertical centerline)", fontweight="bold")
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_line, v_sim, "r-", linewidth=2, label=f"{solver.upper()} N={N}")
    axes[1].scatter(ghia_v["x"], ghia_v["v"], c="black", s=50, marker="x",
                    label="Ghia et al. (1982)", zorder=10)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].set_title("V velocity (horizontal centerline)", fontweight="bold")
    axes[1].legend(frameon=True)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Ghia Benchmark Validation — Re={Re:.0f}", fontweight="bold", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / "ghia_validation.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# =============================================================================
# Comparison Plot (for parent run)
# =============================================================================


def _build_method_label(sibling: dict) -> str:
    """Build a unified method label from solver name.

    Formats solver names nicely for legends:
    - 'fv' -> 'FV'
    - 'spectral' -> 'Spectral'
    - 'spectral_fsg' -> 'Spectral-FSG'
    - 'spectral_vmg' -> 'Spectral-VMG'
    """
    solver = sibling.get("solver", "unknown")

    # Format known solver names
    label_map = {
        "fv": "FV",
        "spectral": "Spectral",
        "spectral_fsg": "Spectral-FSG",
        "spectral_vmg": "Spectral-VMG",
        "spectral_fmg": "Spectral-FMG",
    }

    return label_map.get(solver, solver.replace("_", "-").title())


def plot_ghia_comparison(siblings: list[dict], tracking_uri: str, output_dir: Path) -> Path:
    """Plot Ghia comparison with all sibling runs using seaborn.

    Legend system (native seaborn):
    - hue: Method type (Spectral, FV, Spectral-FSG, etc.)
    - style: Grid size N (different dash patterns)
    - markers: Method-specific markers (every 20th point)

    Parameters
    ----------
    siblings : list[dict]
        List of sibling run info dicts with run_id, N, Re, solver
    tracking_uri : str
        MLflow tracking URI
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to comparison plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not siblings:
        return None

    # Only plot finished runs
    finished_siblings = [s for s in siblings if s.get("status", "FINISHED") == "FINISHED"]
    if len(finished_siblings) < 2:
        log.info(f"Need at least 2 finished runs for comparison (have {len(finished_siblings)})")
        return None

    siblings = finished_siblings
    Re = siblings[0]["Re"]

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]
    if int(Re) not in AVAILABLE_RE:
        log.warning(f"Ghia data not available for Re={Re}")
        return None

    # Load Ghia reference data
    project_root = Path(__file__).parent.parent
    ghia_dir = project_root / "data" / "validation" / "ghia"

    u_file = ghia_dir / f"ghia_Re{int(Re)}_u_centerline.csv"
    v_file = ghia_dir / f"ghia_Re{int(Re)}_v_centerline.csv"

    if not u_file.exists() or not v_file.exists():
        log.warning(f"Ghia data files not found")
        return None

    ghia_u = pd.read_csv(u_file)
    ghia_v = pd.read_csv(v_file)

    # Load scientific style
    style_path = project_root / "src" / "utils" / "plotting" / "scientific.mplstyle"
    if style_path.exists():
        plt.style.use(style_path)
    else:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)

    # Filter to unique (method, N) combinations
    seen_combos = set()
    unique_siblings = []
    for sibling in siblings:
        method = _build_method_label(sibling)
        N = sibling["N"]
        combo = (method, N)
        if combo not in seen_combos:
            seen_combos.add(combo)
            unique_siblings.append(sibling)

    log.info(f"Plotting {len(unique_siblings)} unique (method, N) combinations")

    # Build DataFrames
    u_records = []
    v_records = []

    for sibling in unique_siblings:
        run_id = sibling["run_id"]
        N = sibling["N"]
        method = _build_method_label(sibling)

        try:
            artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
            fields = load_fields_from_zarr(artifact_dir)
            x_unique, y_unique, U_2d, V_2d, _ = restructure_fields(fields)

            from spectral import spectral_interpolate

            n_points = 200
            y_line = np.linspace(y_unique.min(), y_unique.max(), n_points)
            x_line = np.linspace(x_unique.min(), x_unique.max(), n_points)

            x_center_idx = len(x_unique) // 2
            y_center_idx = len(y_unique) // 2

            u_sim = spectral_interpolate(y_unique, U_2d[:, x_center_idx], y_line, basis="legendre")
            v_sim = spectral_interpolate(x_unique, V_2d[y_center_idx, :], x_line, basis="legendre")

            for i in range(n_points):
                u_records.append({"y": y_line[i], "u": u_sim[i], "Method": method, "N": N})
                v_records.append({"x": x_line[i], "v": v_sim[i], "Method": method, "N": N})

        except Exception as e:
            log.warning(f"Failed to load run {run_id}: {e}")
            continue

    if not u_records:
        log.warning("No valid runs to plot")
        return None

    u_df = pd.DataFrame(u_records).sort_values(["Method", "N", "y"])
    v_df = pd.DataFrame(v_records).sort_values(["Method", "N", "x"])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot with seaborn: hue=Method (color), style=N (dashes), markers
    sns.lineplot(
        data=u_df, x="u", y="y",
        hue="Method", style="N",
        markers=True, markersize=4, markevery=20,
        linewidth=1.5, sort=False,
        ax=axes[0],
    )
    axes[0].scatter(ghia_u["u"], ghia_u["y"], c="black", s=60, marker="x",
                    linewidths=1.5, label=r"Ghia et al.\ (1982)", zorder=10)
    axes[0].set_xlabel(r"$u$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_title(r"$u$-velocity (vertical centerline)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_xlim(-0.4, 1.05)
    axes[0].set_ylim(0, 1)

    sns.lineplot(
        data=v_df, x="x", y="v",
        hue="Method", style="N",
        markers=True, markersize=4, markevery=20,
        linewidth=1.5, sort=False,
        ax=axes[1],
    )
    axes[1].scatter(ghia_v["x"], ghia_v["v"], c="black", s=60, marker="x",
                    linewidths=1.5, label=r"Ghia et al.\ (1982)", zorder=10)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$v$")
    axes[1].set_title(r"$v$-velocity (horizontal centerline)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_xlim(0, 1)

    fig.suptitle(rf"Ghia Benchmark Comparison --- $\mathrm{{Re}}={int(Re)}$")
    plt.tight_layout()

    output_path = output_dir / "ghia_comparison.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved comparison plot: {output_path.name}")
    return output_path


# =============================================================================
# MLflow Upload
# =============================================================================


def upload_plots_to_mlflow(run_id: str, plot_paths: list, tracking_uri: str, artifact_subdir: str = "plots"):
    """Upload generated plots to MLflow run as artifacts."""
    mlflow.set_tracking_uri(tracking_uri)

    valid_paths = [p for p in plot_paths if p and p.exists()]

    # Check if we're already in an active run
    active_run = mlflow.active_run()
    if active_run and active_run.info.run_id == run_id:
        # Already in the correct run, just log artifacts
        for path in valid_paths:
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
            log.info(f"Uploaded: {artifact_subdir}/{path.name}")
    else:
        # Start/resume run to upload artifacts
        with mlflow.start_run(run_id=run_id, nested=True):
            for path in valid_paths:
                mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
                log.info(f"Uploaded: {artifact_subdir}/{path.name}")


# =============================================================================
# Direct API for run_solver.py
# =============================================================================


def generate_plots_for_run(
    run_id: str,
    tracking_uri: str,
    output_dir: Path,
    solver_name: str,
    N: int,
    Re: float,
    parent_run_id: Optional[str] = None,
    upload_to_mlflow: bool = True,
) -> list[Path]:
    """Generate all plots for a completed run.

    Called directly from run_solver.py after solver completes.

    Parameters
    ----------
    run_id : str
        MLflow run ID
    tracking_uri : str
        MLflow tracking URI
    output_dir : Path
        Directory to save plots
    solver_name : str
        Solver name (e.g., "spectral", "spectral_fsg", "fv")
    N : int
        Grid size parameter
    Re : float
        Reynolds number
    parent_run_id : str, optional
        Parent run ID if this is part of a sweep
    upload_to_mlflow : bool
        Whether to upload plots to MLflow

    Returns
    -------
    list[Path]
        List of generated plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download artifacts and load data
    artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
    fields = load_fields_from_zarr(artifact_dir)
    fields_df = fields_to_dataframe(fields)
    timeseries_df = load_timeseries_from_mlflow(run_id, tracking_uri)

    log.info(f"Generating plots for {solver_name} N={N} Re={Re}")

    # Generate individual plots
    plot_paths = []
    plot_paths.append(plot_fields(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_streamlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_vorticity(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_centerlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_convergence(timeseries_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_ghia_validation(fields, Re, solver_name, N, output_dir))

    plot_paths = [p for p in plot_paths if p is not None]
    log.info(f"Generated {len(plot_paths)} plots for run")

    # Upload to individual run
    if upload_to_mlflow:
        upload_plots_to_mlflow(run_id, plot_paths, tracking_uri)

    log.info("Plotting done!")
    return plot_paths


def generate_comparison_plots_for_sweep(
    parent_run_ids: list[str],
    tracking_uri: str,
    output_dir: Path,
    upload_to_mlflow: bool = True,
) -> dict[str, Path]:
    """Generate comparison plots for all parent runs after sweep completes.

    Called from MLflow callback's on_multirun_end after all jobs finish.

    Parameters
    ----------
    parent_run_ids : list[str]
        List of parent run IDs (one per Re value)
    tracking_uri : str
        MLflow tracking URI
    output_dir : Path
        Base output directory for comparison plots
    upload_to_mlflow : bool
        Whether to upload plots to MLflow

    Returns
    -------
    dict[str, Path]
        Mapping of parent_run_id to comparison plot path
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for parent_run_id in parent_run_ids:
        log.info(f"Generating comparison plot for parent run: {parent_run_id[:8]}...")

        # Find all children of this parent
        siblings = find_sibling_runs(parent_run_id, tracking_uri)

        if len(siblings) < 2:
            log.warning(f"  Only {len(siblings)} child run(s), skipping comparison")
            continue

        # Check all siblings are finished
        unfinished = [s for s in siblings if s.get("status") != "FINISHED"]
        if unfinished:
            log.warning(f"  {len(unfinished)} run(s) still not finished, skipping")
            continue

        # Get parent run info for naming
        client = mlflow.tracking.MlflowClient()
        parent_run = client.get_run(parent_run_id)
        parent_name = parent_run.info.run_name or parent_run_id[:8]

        # Create comparison plot
        comparison_dir = output_dir / parent_name
        comparison_dir.mkdir(exist_ok=True)

        comparison_path = plot_ghia_comparison(siblings, tracking_uri, comparison_dir)

        if comparison_path:
            results[parent_run_id] = comparison_path
            log.info(f"  Created comparison plot: {comparison_path.name}")

            if upload_to_mlflow:
                upload_plots_to_mlflow(parent_run_id, [comparison_path], tracking_uri, "plots")
                log.info(f"  Uploaded to parent run")

    log.info(f"Generated {len(results)} comparison plot(s)")
    return results


# =============================================================================
# Main Entry Point (Standalone Hydra Usage)
# =============================================================================


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point - finds matching run and generates plots."""

    tracking_uri = cfg.mlflow.get("tracking_uri", "./mlruns")
    solver_name = cfg.solver.name
    N = cfg.N
    Re = cfg.Re

    # Find matching MLflow run
    run_id, parent_run_id = find_matching_run(cfg, tracking_uri)

    # Setup output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download artifacts for this run
    artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
    fields = load_fields_from_zarr(artifact_dir)
    fields_df = fields_to_dataframe(fields)
    timeseries_df = load_timeseries_from_mlflow(run_id, tracking_uri)

    log.info(f"Generating plots for {solver_name} N={N} Re={Re}")

    # ==========================================================================
    # Individual run plots
    # ==========================================================================
    plot_paths = []

    plot_paths.append(plot_fields(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_streamlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_vorticity(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_centerlines(fields_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_convergence(timeseries_df, Re, solver_name, N, output_dir))
    plot_paths.append(plot_ghia_validation(fields, Re, solver_name, N, output_dir))

    plot_paths = [p for p in plot_paths if p is not None]
    log.info(f"Generated {len(plot_paths)} plots for run")

    # Upload to individual run
    if cfg.get("upload_to_mlflow", True):
        upload_plots_to_mlflow(run_id, plot_paths, tracking_uri)

    # ==========================================================================
    # Comparison plot for parent (if this is part of a sweep)
    # ==========================================================================
    if parent_run_id:
        log.info("This run is part of a sweep - generating comparison plot for parent")

        siblings = find_sibling_runs(parent_run_id, tracking_uri)

        if len(siblings) > 1:
            comparison_dir = output_dir / "comparison"
            comparison_dir.mkdir(exist_ok=True)

            comparison_path = plot_ghia_comparison(siblings, tracking_uri, comparison_dir)

            if comparison_path and cfg.get("upload_to_mlflow", True):
                upload_plots_to_mlflow(parent_run_id, [comparison_path], tracking_uri, "plots")
                log.info("Comparison plot uploaded to parent run")

    log.info("Done!")


if __name__ == "__main__":
    main()
