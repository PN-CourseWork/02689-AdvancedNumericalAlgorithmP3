"""
Field Visualization Plots for LDC.

Generates contour plots for pressure/velocity fields,
streamlines, and vorticity.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import RectBivariateSpline

log = logging.getLogger(__name__)


def plot_fields(
    fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path
) -> Path:
    """Generate field contour plots (p, u, v)."""
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    cf_p = axes[0].contourf(X_fine, Y_fine, P_interp, levels=30, cmap="viridis")
    axes[0].set_xlabel(r"$x$", fontsize=11)
    axes[0].set_ylabel(r"$y$", fontsize=11)
    axes[0].set_title(r"\textbf{Pressure}", fontsize=12)
    axes[0].set_aspect("equal")
    cbar_p = plt.colorbar(cf_p, ax=axes[0], label=r"$p$")
    cbar_p.ax.tick_params(labelsize=9)

    cf_u = axes[1].contourf(X_fine, Y_fine, U_interp, levels=30, cmap="RdBu_r")
    axes[1].set_xlabel(r"$x$", fontsize=11)
    axes[1].set_ylabel(r"$y$", fontsize=11)
    axes[1].set_title(r"\textbf{$u$-velocity}", fontsize=12)
    axes[1].set_aspect("equal")
    cbar_u = plt.colorbar(cf_u, ax=axes[1], label=r"$u$")
    cbar_u.ax.tick_params(labelsize=9)

    cf_v = axes[2].contourf(X_fine, Y_fine, V_interp, levels=30, cmap="RdBu_r")
    axes[2].set_xlabel(r"$x$", fontsize=11)
    axes[2].set_ylabel(r"$y$", fontsize=11)
    axes[2].set_title(r"\textbf{$v$-velocity}", fontsize=12)
    axes[2].set_aspect("equal")
    cbar_v = plt.colorbar(cf_v, ax=axes[2], label=r"$v$")
    cbar_v.ax.tick_params(labelsize=9)

    solver_label = solver.upper().replace("_", r"\_")
    fig.suptitle(
        rf"\textbf{{Solution Fields}} --- {solver_label}, $N={N}$, $\mathrm{{Re}}={Re:.0f}$",
        fontsize=13,
        y=1.00,
    )

    plt.tight_layout()

    output_path = output_dir / "fields.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_streamlines(
    fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path
) -> Path:
    """Generate streamline plot with velocity magnitude."""
    x_unique = np.sort(fields_df["x"].unique())
    y_unique = np.sort(fields_df["y"].unique())
    nx, ny = len(x_unique), len(y_unique)

    sorted_df = fields_df.sort_values(["y", "x"])
    U = sorted_df["u"].values.reshape(ny, nx)
    V = sorted_df["v"].values.reshape(ny, nx)

    n_fine = 250
    x_fine = np.linspace(x_unique[0], x_unique[-1], n_fine)
    y_fine = np.linspace(y_unique[0], y_unique[-1], n_fine)

    U_interp = RectBivariateSpline(y_unique, x_unique, U)(y_fine, x_fine)
    V_interp = RectBivariateSpline(y_unique, x_unique, V)(y_fine, x_fine)
    vel_mag = np.sqrt(U_interp**2 + V_interp**2)

    fig, ax = plt.subplots(figsize=(8, 7))

    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Smooth contours with coolwarm colormap
    cf = ax.contourf(X_fine, Y_fine, vel_mag, levels=40, cmap="coolwarm")

    # Semi-transparent white streamlines (RGBA for transparency)
    stream = ax.streamplot(
        x_fine,
        y_fine,
        U_interp,
        V_interp,
        density=2.0,
        linewidth=1.5,
        arrowsize=1.3,
        arrowstyle="->",
        color=(1, 1, 1, 0.7),  # RGBA white with 70% opacity
        zorder=2,
    )

    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    solver_label = solver.upper().replace("_", r" ")
    ax.set_title(
        rf"Streamlines: {solver_label}, $N={N}$, $\mathrm{{Re}}={Re:.0f}$",
        fontsize=13,
    )
    ax.set_aspect("equal")

    # Horizontal colorbar at bottom
    cbar = plt.colorbar(
        cf,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        aspect=30,
        label=r"Velocity Magnitude $|\mathbf{u}|$",
    )
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    output_path = output_dir / "streamlines.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_vorticity(
    fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path
) -> Path:
    """Generate vorticity contour plot."""
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

    fig, ax = plt.subplots(figsize=(7, 6))

    vmax = np.max(np.abs(vorticity))
    cf = ax.contourf(
        X_fine, Y_fine, vorticity, levels=30, vmin=-vmax, vmax=vmax, cmap="RdBu_r"
    )

    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$y$", fontsize=11)
    solver_label = solver.upper().replace("_", r"\_")
    ax.set_title(
        rf"\textbf{{Vorticity}} --- {solver_label}, $N={N}$, $\mathrm{{Re}}={Re:.0f}$",
        fontsize=12,
    )
    ax.set_aspect("equal")
    cbar = plt.colorbar(
        cf, ax=ax, label=r"$\omega = \partial v/\partial x - \partial u/\partial y$"
    )
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    output_path = output_dir / "vorticity.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_streamlines_pyvista(
    fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path
) -> Path:
    """Generate beautiful streamline plot using PyVista with ParaView theme.

    Creates a visually striking visualization with velocity magnitude field
    and streamlines, with transparent background for easy compositing.
    """
    x_unique = np.sort(fields_df["x"].unique())
    y_unique = np.sort(fields_df["y"].unique())
    nx, ny = len(x_unique), len(y_unique)

    sorted_df = fields_df.sort_values(["y", "x"])
    U = sorted_df["u"].values.reshape(ny, nx)
    V = sorted_df["v"].values.reshape(ny, nx)

    # Interpolate to finer grid for smoother visualization
    n_fine = 200
    x_fine = np.linspace(x_unique[0], x_unique[-1], n_fine)
    y_fine = np.linspace(y_unique[0], y_unique[-1], n_fine)

    U_interp = RectBivariateSpline(y_unique, x_unique, U)(y_fine, x_fine)
    V_interp = RectBivariateSpline(y_unique, x_unique, V)(y_fine, x_fine)

    # Create 3D grid (z=0 plane)
    X, Y = np.meshgrid(x_fine, y_fine)
    Z = np.zeros_like(X)

    # Create structured grid
    grid = pv.StructuredGrid(X, Y, Z)

    # Add velocity as vector field (3D with w=0)
    vel_mag = np.sqrt(U_interp**2 + V_interp**2)
    vectors = np.zeros((n_fine * n_fine, 3))
    vectors[:, 0] = U_interp.ravel()
    vectors[:, 1] = V_interp.ravel()
    vectors[:, 2] = 0.0

    grid["velocity"] = vectors
    grid["velocity_magnitude"] = vel_mag.ravel()
    grid.set_active_vectors("velocity")

    # Create seed points for streamlines
    n_seeds = 15
    seed_points_left = np.column_stack([
        np.full(n_seeds, x_fine[3]),
        np.linspace(y_fine[3], y_fine[-4], n_seeds),
        np.zeros(n_seeds)
    ])
    seed_points_bottom = np.column_stack([
        np.linspace(x_fine[3], x_fine[-4], n_seeds),
        np.full(n_seeds, y_fine[3]),
        np.zeros(n_seeds)
    ])
    seed_points = np.vstack([seed_points_left, seed_points_bottom])
    seeds = pv.PolyData(seed_points)

    # Generate streamlines
    streamlines = grid.streamlines_from_source(
        seeds,
        vectors="velocity",
        max_steps=2000,
        integration_direction="both",
    )

    # Set up PyVista plotter with ParaView theme
    pv.set_plot_theme("paraview")
    plotter = pv.Plotter(off_screen=True, window_size=[1400, 1200])

    # Add velocity magnitude field as background surface
    surface = grid.extract_surface()
    plotter.add_mesh(
        surface,
        scalars="velocity_magnitude",
        cmap="turbo",
        show_edges=False,
        lighting=False,
        scalar_bar_args={
            "title": "Velocity Magnitude |u|",
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.2,
            "width": 0.08,
            "height": 0.6,
            "title_font_size": 16,
            "label_font_size": 14,
            "fmt": "%.3f",
            "n_labels": 5,
        },
    )

    # Add streamlines as white tubes on top
    if streamlines.n_points > 0:
        tubes = streamlines.tube(radius=0.004, n_sides=8)
        plotter.add_mesh(
            tubes,
            color="white",
            opacity=0.85,
            smooth_shading=True,
            specular=0.3,
        )

    # Set up camera for clean 2D view
    plotter.camera_position = "xy"
    plotter.camera.zoom(1.15)

    # Enable anti-aliasing for smooth edges
    plotter.enable_anti_aliasing("ssaa")

    # Save with transparent background
    output_path = output_dir / "streamlines_3d.png"
    plotter.screenshot(output_path, transparent_background=True, scale=2)
    plotter.close()

    return output_path


def export_fields_to_vtk(
    fields_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path
) -> Path:
    """Export solution fields to VTK format for ParaView visualization.

    Creates a structured grid VTK file with pressure, velocity components,
    velocity magnitude, and vorticity fields.
    """
    x_unique = np.sort(fields_df["x"].unique())
    y_unique = np.sort(fields_df["y"].unique())
    nx, ny = len(x_unique), len(y_unique)

    sorted_df = fields_df.sort_values(["y", "x"])
    P = sorted_df["p"].values.reshape(ny, nx)
    U = sorted_df["u"].values.reshape(ny, nx)
    V = sorted_df["v"].values.reshape(ny, nx)

    # Create 3D grid (z=0 plane)
    X, Y = np.meshgrid(x_unique, y_unique)
    Z = np.zeros_like(X)

    # Create structured grid
    grid = pv.StructuredGrid(X, Y, Z)

    # Add scalar fields
    grid["pressure"] = P.ravel()
    grid["u"] = U.ravel()
    grid["v"] = V.ravel()
    grid["velocity_magnitude"] = np.sqrt(U**2 + V**2).ravel()

    # Compute and add vorticity
    U_spline = RectBivariateSpline(y_unique, x_unique, U)
    V_spline = RectBivariateSpline(y_unique, x_unique, V)
    dvdx = V_spline(y_unique, x_unique, dx=1)
    dudy = U_spline(y_unique, x_unique, dy=1)
    vorticity = dvdx - dudy
    grid["vorticity"] = vorticity.ravel()

    # Add velocity vector field
    vectors = np.zeros((nx * ny, 3))
    vectors[:, 0] = U.ravel()
    vectors[:, 1] = V.ravel()
    grid["velocity"] = vectors

    # Save as VTK
    output_path = output_dir / "solution.vtk"
    grid.save(output_path)

    return output_path
