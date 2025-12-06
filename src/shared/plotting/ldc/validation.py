"""
Validation and Comparison Plots for LDC.

Generates centerline velocity profiles and Ghia benchmark comparisons.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import RectBivariateSpline

from spectral import spectral_interpolate
from utilities.config.paths import get_repo_root

from .data_loading import load_fields_from_zarr, restructure_fields
from .mlflow_utils import download_mlflow_artifacts

log = logging.getLogger(__name__)


def plot_centerlines(
    fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path
) -> Path:
    """Plot velocity profiles along centerlines."""
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

    axes[0].plot(u_vertical, y_line, linewidth=2)
    axes[0].set_xlabel(r"$u$", fontsize=11)
    axes[0].set_ylabel(r"$y$", fontsize=11)
    axes[0].set_title(r"\textbf{$u$-velocity along vertical centerline}", fontsize=12)
    axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    axes[1].plot(x_line, v_horizontal, linewidth=2)
    axes[1].set_xlabel(r"$x$", fontsize=11)
    axes[1].set_ylabel(r"$v$", fontsize=11)
    axes[1].set_title(r"\textbf{$v$-velocity along horizontal centerline}", fontsize=12)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    solver_label = solver.upper().replace("_", r"\_")
    fig.suptitle(
        rf"\textbf{{Centerline Profiles}} --- {solver_label}, $N={N}$, $\mathrm{{Re}}={Re:.0f}$",
        fontsize=13,
        y=0.98,
    )

    plt.tight_layout()

    output_path = output_dir / "centerlines.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


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


def plot_ghia_comparison(
    siblings: list[dict], tracking_uri: str, output_dir: Path
) -> Path:
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
    if not siblings:
        return None

    # Only plot finished runs
    finished_siblings = [
        s for s in siblings if s.get("status", "FINISHED") == "FINISHED"
    ]
    if len(finished_siblings) < 1:
        log.info(
            f"Need at least 1 finished run for comparison (have {len(finished_siblings)})"
        )
        return None

    siblings = finished_siblings
    Re = siblings[0]["Re"]

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]
    if int(Re) not in AVAILABLE_RE:
        log.warning(f"Ghia data not available for Re={Re}")
        return None

    # Load Ghia reference data from repo-root data/validation/ghia
    project_root = get_repo_root()
    ghia_dir = project_root / "data" / "validation" / "ghia"

    u_file = ghia_dir / f"ghia_Re{int(Re)}_u_centerline.csv"
    v_file = ghia_dir / f"ghia_Re{int(Re)}_v_centerline.csv"

    if not u_file.exists() or not v_file.exists():
        raise FileNotFoundError(f"Ghia data files not found in {ghia_dir}")

    ghia_u = pd.read_csv(u_file)
    ghia_v = pd.read_csv(v_file)

    # Filter to unique (solver/method, N) combinations
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
        solver_name = sibling.get("solver", method)

        try:
            artifact_dir = download_mlflow_artifacts(run_id, tracking_uri)
            fields = load_fields_from_zarr(artifact_dir)
            x_unique, y_unique, U_2d, V_2d, _ = restructure_fields(fields)

            n_points = 200
            y_line = np.linspace(y_unique.min(), y_unique.max(), n_points)
            x_line = np.linspace(x_unique.min(), x_unique.max(), n_points)

            # Find physical center (x=0.5, y=0.5), not middle index!
            # For non-uniform grids (Chebyshev), middle index != physical center
            x_center = 0.5 * (x_unique.min() + x_unique.max())
            y_center = 0.5 * (y_unique.min() + y_unique.max())
            x_center_idx = np.argmin(np.abs(x_unique - x_center))
            y_center_idx = np.argmin(np.abs(y_unique - y_center))

            # Use appropriate interpolation based on solver type
            # FV uses uniform grids -> linear interpolation
            # Spectral uses non-uniform grids -> spectral interpolation
            if solver_name.lower().startswith("fv"):
                # Linear interpolation for FV solvers
                u_sim = np.interp(y_line, y_unique, U_2d[:, x_center_idx])
                v_sim = np.interp(x_line, x_unique, V_2d[y_center_idx, :])
            else:
                # Spectral interpolation for spectral solvers
                u_sim = spectral_interpolate(
                    y_unique, U_2d[:, x_center_idx], y_line, basis="legendre"
                )
                v_sim = spectral_interpolate(
                    x_unique, V_2d[y_center_idx, :], x_line, basis="legendre"
                )

            # Create combined method-N label with LaTeX formatting
            method_label = f"{method}, $N={N}$"

            for i in range(n_points):
                u_records.append(
                    {
                        "y": y_line[i],
                        "u": u_sim[i],
                        "Method": method_label,
                        "Solver": solver_name,
                        "N": N,
                    }
                )
                v_records.append(
                    {
                        "x": x_line[i],
                        "v": v_sim[i],
                        "Method": method_label,
                        "Solver": solver_name,
                        "N": N,
                    }
                )

        except Exception as e:
            log.warning(f"Failed to load run {run_id}: {e}")
            continue

    if not u_records:
        log.warning("No valid runs to plot")
        return None

    u_df = pd.DataFrame(u_records)
    v_df = pd.DataFrame(v_records)

    # Create subplots with publication-quality sizing
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: u-velocity (vertical centerline)
    sns.lineplot(
        data=u_df,
        x="u",
        y="y",
        hue="Method",
        style="Method",
        markers=True,
        ax=axes[0],
        sort=False,
        linewidth=2,
        markersize=7,
        markevery=15,
    )
    sns.scatterplot(
        data=ghia_u,
        x="u",
        y="y",
        marker="o",
        s=50,
        facecolors="none",
        edgecolors="#333333",
        linewidths=1.2,
        label="Ghia et al. (1982)",
        ax=axes[0],
        zorder=10,
    )
    axes[0].set_xlabel(r"$u$", fontsize=11)
    axes[0].set_ylabel(r"$y$", fontsize=11)
    axes[0].set_title(r"$u$-velocity (vertical centerline)", fontsize=11)

    # Right: v-velocity (horizontal centerline)
    sns.lineplot(
        data=v_df,
        x="x",
        y="v",
        hue="Method",
        style="Method",
        markers=True,
        ax=axes[1],
        sort=False,
        linewidth=2,
        markersize=7,
        markevery=15,
    )
    sns.scatterplot(
        data=ghia_v,
        x="x",
        y="v",
        marker="o",
        s=50,
        facecolors="none",
        edgecolors="#333333",
        linewidths=1.2,
        label="Ghia et al. (1982)",
        ax=axes[1],
        zorder=10,
    )
    axes[1].set_xlabel(r"$x$", fontsize=11)
    axes[1].set_ylabel(r"$v$", fontsize=11)
    axes[1].set_title(r"$v$-velocity (horizontal centerline)", fontsize=11)

    # Overall title
    fig.suptitle(rf"Ghia Benchmark Comparison ($\mathrm{{Re}} = {int(Re)}$)", fontsize=13, y=1.00)

    # Tight layout for better spacing
    plt.tight_layout()

    output_path = output_dir / "ghia_comparison.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved comparison plot: {output_path.name}")
    return output_path
