"""
Solution Plotter - Automatically finds and plots MLflow runs matching config.

Uses the same Hydra config as run_solver.py to find matching runs.

For individual runs: generates fields, streamlines, vorticity, centerlines,
                     ghia_validation, and convergence plots.

For sweeps: additionally generates a Ghia comparison plot uploaded to parent run.

Usage:
    # Plot single run
    uv run python Experiments/Solution/plot_solution.py solver=spectral N=31 Re=100

    # Plot sweep (comparison plot goes to parent)
    uv run python Experiments/Solution/plot_solution.py -m solver=spectral N=15,31,63 Re=100

    # Use remote MLflow
    uv run python Experiments/Solution/plot_solution.py solver=spectral N=31 Re=100 mlflow=coolify
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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

    # Find all children of this parent
    filter_string = f"tags.parent_run_id = '{parent_run_id}' AND attributes.status = 'FINISHED'"

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["params.nx ASC"],  # Sort by N for nice legend order
        max_results=50,
    )

    siblings = []
    for run in runs:
        # Extract solver name from run_name (format: {solver}_N{n}_Re{re})
        run_name = run.info.run_name or ""
        solver_name = run_name.split("_N")[0] if "_N" in run_name else "unknown"
        siblings.append({
            "run_id": run.info.run_id,
            "run_name": run_name,
            "N": int(run.data.params.get("nx", 0)),
            "Re": float(run.data.params.get("Re", 0)),
            "solver": solver_name,
        })

    log.info(f"Found {len(siblings)} sibling runs in sweep")
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

    output_path = output_dir / "fields.png"
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

    output_path = output_dir / "streamlines.png"
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

    output_path = output_dir / "vorticity.png"
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

    output_path = output_dir / "centerlines.png"
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

    output_path = output_dir / "convergence.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_ghia_validation(fields: dict, Re: float, solver: str, N: int, output_dir: Path) -> Path:
    """Plot validation against Ghia benchmark data."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]
    if int(Re) not in AVAILABLE_RE:
        log.warning(f"Ghia data not available for Re={Re}. Available: {AVAILABLE_RE}")
        return None

    # Project root is 2 levels up from this file
    project_root = Path(__file__).parent.parent.parent
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

    U_spline = RectBivariateSpline(y_unique, x_unique, U_2d)
    V_spline = RectBivariateSpline(y_unique, x_unique, V_2d)

    n_points = 200
    y_line = np.linspace(y_unique.min(), y_unique.max(), n_points)
    x_line = np.linspace(x_unique.min(), x_unique.max(), n_points)

    x_center = (x_unique.min() + x_unique.max()) / 2
    y_center = (y_unique.min() + y_unique.max()) / 2

    u_sim = U_spline(y_line, x_center).ravel()
    v_sim = V_spline(y_center, x_line).ravel()

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

    output_path = output_dir / "ghia_validation.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# =============================================================================
# Comparison Plot (for parent run)
# =============================================================================


def plot_ghia_comparison(siblings: list[dict], tracking_uri: str, output_dir: Path) -> Path:
    """Plot Ghia comparison with all sibling runs overlaid.

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
    from scipy.interpolate import RectBivariateSpline

    if not siblings:
        return None

    # Get Re from first sibling (all should have same Re in a sweep)
    Re = siblings[0]["Re"]

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]
    if int(Re) not in AVAILABLE_RE:
        log.warning(f"Ghia data not available for Re={Re}")
        return None

    # Project root is 2 levels up from this file
    project_root = Path(__file__).parent.parent.parent
    ghia_dir = project_root / "data" / "validation" / "ghia"

    u_file = ghia_dir / f"ghia_Re{int(Re)}_u_centerline.csv"
    v_file = ghia_dir / f"ghia_Re{int(Re)}_v_centerline.csv"

    if not u_file.exists() or not v_file.exists():
        log.warning(f"Ghia data files not found")
        return None

    ghia_u = pd.read_csv(u_file)
    ghia_v = pd.read_csv(v_file)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color map and line styles for different runs
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(siblings)))
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # Solid, dashed, dashdot, dotted, custom

    for i, sibling in enumerate(siblings):
        run_id = sibling["run_id"]
        N = sibling["N"]
        solver = sibling["solver"]

        try:
            # Download and load fields
            artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
            fields = load_fields_from_zarr(artifact_dir)

            # Restructure to 2D arrays
            x_unique, y_unique, U_2d, V_2d, _ = restructure_fields(fields)

            U_spline = RectBivariateSpline(y_unique, x_unique, U_2d)
            V_spline = RectBivariateSpline(y_unique, x_unique, V_2d)

            n_points = 200
            y_line = np.linspace(y_unique.min(), y_unique.max(), n_points)
            x_line = np.linspace(x_unique.min(), x_unique.max(), n_points)

            x_center = (x_unique.min() + x_unique.max()) / 2
            y_center = (y_unique.min() + y_unique.max()) / 2

            u_sim = U_spline(y_line, x_center).ravel()
            v_sim = V_spline(y_center, x_line).ravel()

            label = f"{solver.upper()} N={N}"
            ls = linestyles[i % len(linestyles)]
            axes[0].plot(u_sim, y_line, color=colors[i], linewidth=2.5, linestyle=ls, label=label)
            axes[1].plot(x_line, v_sim, color=colors[i], linewidth=2.5, linestyle=ls, label=label)

        except Exception as e:
            log.warning(f"Failed to load run {run_id}: {e}")
            continue

    # Add Ghia reference
    axes[0].scatter(ghia_u["u"], ghia_u["y"], c="black", s=60, marker="x",
                    label="Ghia et al. (1982)", zorder=10)
    axes[0].set_xlabel("u", fontsize=12)
    axes[0].set_ylabel("y", fontsize=12)
    axes[0].set_title("U velocity (vertical centerline)", fontweight="bold")
    axes[0].legend(frameon=True, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ghia_v["x"], ghia_v["v"], c="black", s=60, marker="x",
                    label="Ghia et al. (1982)", zorder=10)
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("v", fontsize=12)
    axes[1].set_title("V velocity (horizontal centerline)", fontweight="bold")
    axes[1].legend(frameon=True, loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Ghia Benchmark Comparison — Re={Re:.0f}", fontweight="bold", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / "ghia_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved comparison plot: {output_path}")
    return output_path


# =============================================================================
# MLflow Upload
# =============================================================================


def upload_plots_to_mlflow(run_id: str, plot_paths: list, tracking_uri: str, artifact_subdir: str = "plots"):
    """Upload generated plots to MLflow run as artifacts."""
    mlflow.set_tracking_uri(tracking_uri)

    valid_paths = [p for p in plot_paths if p and p.exists()]

    with mlflow.start_run(run_id=run_id):
        for path in valid_paths:
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
            log.info(f"Uploaded: {artifact_subdir}/{path.name}")


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
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
