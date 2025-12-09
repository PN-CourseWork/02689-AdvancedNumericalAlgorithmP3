"""
Validation and Comparison Plots for LDC.

Generates Ghia benchmark comparisons and L2 convergence plots.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from solvers.spectral.basis import spectral_interpolate
from utilities.config.paths import get_repo_root

from .data_loading import load_fields_from_zarr, restructure_fields
from .mlflow_utils import download_mlflow_artifacts

log = logging.getLogger(__name__)


def plot_l2_convergence(
    siblings: list[dict],
    tracking_uri: str,
    output_dir: Path,
) -> list[Path]:
    """Plot L2 error convergence (log-log) for parent runs.

    Creates 4 separate PDF files:
    - l2_convergence_u.pdf: u-velocity L2 error vs N
    - l2_convergence_v.pdf: v-velocity L2 error vs N
    - l2_convergence_u_regu.pdf: u-velocity L2 error (regularized ref) vs N
    - l2_convergence_v_regu.pdf: v-velocity L2 error (regularized ref) vs N

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
    list[Path]
        Paths to generated convergence plots
    """
    if not siblings:
        return []

    # Only use finished runs
    finished_siblings = [
        s for s in siblings if s.get("status", "FINISHED") == "FINISHED"
    ]
    if len(finished_siblings) < 2:
        log.info(
            f"Need at least 2 finished runs for convergence plot (have {len(finished_siblings)})"
        )
        return []

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Collect L2 error metrics from all runs
    records = []
    for sibling in finished_siblings:
        run_id = sibling["run_id"]
        N = sibling["N"]
        solver = sibling.get("solver", "unknown")
        method = _build_method_label(sibling)

        try:
            run = client.get_run(run_id)
            metrics = run.data.metrics

            # Get L2 error metrics (may not all be present)
            u_l2 = metrics.get("u_L2_error")
            v_l2 = metrics.get("v_L2_error")
            u_l2_regu = metrics.get("u_L2_error_regu")
            v_l2_regu = metrics.get("v_L2_error_regu")

            if u_l2 is not None or v_l2 is not None:
                records.append(
                    {
                        "N": N,
                        "Method": method,
                        "Solver": solver,
                        "u_L2_error": u_l2,
                        "v_L2_error": v_l2,
                        "u_L2_error_regu": u_l2_regu,
                        "v_L2_error_regu": v_l2_regu,
                    }
                )
        except Exception as e:
            log.warning(f"Failed to load metrics for run {run_id}: {e}")
            continue

    if not records:
        log.warning("No L2 error metrics found in sibling runs")
        return []

    df = pd.DataFrame(records)

    # Define the 4 plots to create
    plot_configs = [
        ("u_L2_error", r"$u$ L2 Error", "l2_convergence_u.pdf"),
        ("v_L2_error", r"$v$ L2 Error", "l2_convergence_v.pdf"),
        ("u_L2_error_regu", r"$u$ L2 Error (regularized ref)", "l2_convergence_u_regu.pdf"),
        ("v_L2_error_regu", r"$v$ L2 Error (regularized ref)", "l2_convergence_v_regu.pdf"),
    ]

    output_paths = []
    for metric_col, ylabel, filename in plot_configs:
        # Filter to rows with valid data for this metric
        plot_df = df[df[metric_col].notna()].copy()
        if plot_df.empty:
            log.info(f"No data for {metric_col}, skipping")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        # Seaborn lineplot with markers
        sns.lineplot(
            data=plot_df,
            x="N",
            y=metric_col,
            hue="Method",
            style="Method",
            markers=True,
            ax=ax,
            linewidth=2,
            markersize=8,
        )

        # Reference convergence lines (N^-2 and N^-4 for spectral)
        N_vals = np.array(sorted(plot_df["N"].unique()))
        if len(N_vals) >= 2:
            # Get representative error for reference line placement
            mid_N = N_vals[len(N_vals) // 2]
            mid_err = plot_df[plot_df["N"] == mid_N][metric_col].mean()
            if mid_err > 0:
                # N^-2 reference
                ref_n2 = mid_err * (mid_N / N_vals) ** 2
                ax.plot(
                    N_vals,
                    ref_n2,
                    "--",
                    color="gray",
                    alpha=0.6,
                    linewidth=1.5,
                    label=r"$\mathcal{O}(N^{-2})$",
                )
                # N^-4 reference (faster convergence)
                ref_n4 = mid_err * (mid_N / N_vals) ** 4
                ax.plot(
                    N_vals,
                    ref_n4,
                    ":",
                    color="gray",
                    alpha=0.6,
                    linewidth=1.5,
                    label=r"$\mathcal{O}(N^{-4})$",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N$ (polynomial order)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"L2 Error Convergence", fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", ls="-", alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        log.info(f"Saved L2 convergence plot: {output_path.name}")
        output_paths.append(output_path)

    return output_paths


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

    # Add zoomed inset for v-velocity (right plot) - focus on peak region
    # Find interesting region: near the maximum v value
    v_max_idx = ghia_v["v"].idxmax()
    v_max_x = ghia_v.loc[v_max_idx, "x"]
    v_max_val = ghia_v.loc[v_max_idx, "v"]

    # Create inset axes for v-velocity zoom
    ax1 = axes[1]

    # Get the actual axis limits after autoscaling to compute aspect ratio
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    aspect_ratio = x_range / y_range

    # Position inset in lower-left corner with aspect ratio matching main plot
    # [left, bottom, width, height] in axes fraction
    inset_height = 0.40
    inset_width = inset_height * aspect_ratio
    # Ensure inset stays within subplot (max width ~0.5 to leave room)
    if inset_width > 0.50:
        inset_width = 0.50
        inset_height = inset_width / aspect_ratio
    axins_v = ax1.inset_axes([0.05, 0.05, inset_width, inset_height])

    # Replot data in inset
    sns.lineplot(
        data=v_df,
        x="x",
        y="v",
        hue="Method",
        style="Method",
        markers=True,
        ax=axins_v,
        sort=False,
        linewidth=1.5,
        markersize=5,
        markevery=5,
        legend=False,
    )
    sns.scatterplot(
        data=ghia_v,
        x="x",
        y="v",
        marker="o",
        s=40,
        facecolors="none",
        edgecolors="#333333",
        linewidths=1.0,
        ax=axins_v,
        legend=False,
    )

    # Set zoom region around maximum v with aspect ratio matching main plot
    zoom_y_half = 0.15  # Half-height of zoom region
    zoom_x_half = zoom_y_half * aspect_ratio  # Half-width scaled by aspect ratio

    # Clamp zoom region to stay within axis bounds
    zoom_x_min = max(x_lim[0], v_max_x - zoom_x_half)
    zoom_x_max = min(x_lim[1], v_max_x + zoom_x_half)
    zoom_y_min = max(y_lim[0], v_max_val - zoom_y_half)
    zoom_y_max = min(y_lim[1], v_max_val + zoom_y_half)

    # Adjust to maintain aspect ratio if clamped
    actual_x_range = zoom_x_max - zoom_x_min
    actual_y_range = zoom_y_max - zoom_y_min
    actual_aspect = actual_x_range / actual_y_range

    if actual_aspect > aspect_ratio:
        # x range too wide, shrink it
        new_x_range = actual_y_range * aspect_ratio
        x_center = (zoom_x_min + zoom_x_max) / 2
        zoom_x_min = x_center - new_x_range / 2
        zoom_x_max = x_center + new_x_range / 2
    elif actual_aspect < aspect_ratio:
        # y range too tall, shrink it
        new_y_range = actual_x_range / aspect_ratio
        y_center = (zoom_y_min + zoom_y_max) / 2
        zoom_y_min = y_center - new_y_range / 2
        zoom_y_max = y_center + new_y_range / 2

    axins_v.set_xlim(zoom_x_min, zoom_x_max)
    axins_v.set_ylim(zoom_y_min, zoom_y_max)
    axins_v.set_xlabel("")
    axins_v.set_ylabel("")
    axins_v.set_xticks([])
    axins_v.set_yticks([])

    # Draw box and connecting lines
    mark_inset(axes[1], axins_v, loc1=1, loc2=3, fc="none", ec="0.5", lw=0.8)

    # Overall title
    fig.suptitle(rf"Ghia Benchmark Comparison ($\mathrm{{Re}} = {int(Re)}$)", fontsize=13, y=1.00)

    # Tight layout for better spacing
    plt.tight_layout()

    output_path = output_dir / "ghia_comparison.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved comparison plot: {output_path.name}")
    return output_path
