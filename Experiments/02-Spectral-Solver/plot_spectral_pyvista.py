"""
Spectral Solver Visualization with PyVista
===========================================

This script visualizes the spectral solver solution using PyVista
with a ParaView-inspired theme for publication-quality figures.
"""

# %%
# Imports and Setup
# -----------------

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import BarycentricInterpolator

from utils import get_project_root

# Use ParaView theme
pv.set_plot_theme("paraview")

# %%
# Load Solution from HDF5
# -----------------------

project_root = get_project_root()
fig_dir = project_root / "figures" / "Spectral-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

# Configuration
Re = 100
N = 25  # Polynomial order (N+1 nodes per direction)

# Load from pre-computed HDF5 file
data_file = (
    project_root / "data" / "Spectral-Solver" / "Chebyshev" / f"LDC_N{N}_Re{Re}.h5"
)

if not data_file.exists():
    raise FileNotFoundError(
        f"Data file not found: {data_file}\n"
        f"Run compute_spectral_chebyshev.py first to generate the data."
    )

print(f"Loading spectral solution from: {data_file}")

with pd.HDFStore(data_file, "r") as store:
    params = store["params"].iloc[0]
    metrics = store["metrics"].iloc[0]
    fields_df = store["fields"]

# Infer actual grid size from data
n_points = len(fields_df)
grid_size = int(np.sqrt(n_points))

print(f"\nSolution loaded: Re={params['Re']:.0f}, Grid={grid_size}x{grid_size}")
print(f"  Converged: {metrics['converged']}")
print(f"  Iterations: {int(metrics['iterations'])}")
print(f"  Final residual: {metrics['final_residual']:.2e}")
print(f"  Wall time: {metrics['wall_time_seconds']:.2f} seconds")

# %%
# Interpolate to Uniform Grid (Barycentric Interpolation)
# -------------------------------------------------------
# Spectral methods use non-uniform LGL nodes clustered near boundaries.
# For smooth visualization and proper streamlines, interpolate to uniform grid.

# Get original solution data from loaded DataFrame
x_orig = fields_df["x"].values
y_orig = fields_df["y"].values
u_orig = fields_df["u"].values
v_orig = fields_df["v"].values
p_orig = fields_df["p"].values

# Infer grid dimensions from the data (assume square grid)
n_points = len(x_orig)
nx_orig = ny_orig = int(np.sqrt(n_points))
if nx_orig * ny_orig != n_points:
    raise ValueError(f"Cannot infer square grid from {n_points} points")

# Get unique LGL node coordinates
x_lgl = np.sort(np.unique(x_orig))
y_lgl = np.sort(np.unique(y_orig))

# Reshape to 2D (assuming row-major ordering from spectral solver)
# Need to figure out the ordering - let's sort and reshape
sorted_idx = np.lexsort((x_orig, y_orig))
U_orig_2d = u_orig[sorted_idx].reshape(ny_orig, nx_orig)
V_orig_2d = v_orig[sorted_idx].reshape(ny_orig, nx_orig)
P_orig_2d = p_orig[sorted_idx].reshape(ny_orig, nx_orig)

# Interpolation resolution (uniform grid)
interp_resolution = 100

# Create fine uniform grid
x_fine = np.linspace(0, 1, interp_resolution)
y_fine = np.linspace(0, 1, interp_resolution)


# Tensor product barycentric interpolation
def interp_2d_barycentric(field_2d, x_nodes, y_nodes, x_new, y_new):
    """Interpolate 2D field using tensor product barycentric interpolation."""
    nx_new = len(x_new)
    # First interpolate along x for each y
    temp = np.array([BarycentricInterpolator(x_nodes, row)(x_new) for row in field_2d])
    # Then interpolate along y for each x
    result = np.array(
        [BarycentricInterpolator(y_nodes, temp[:, i])(y_new) for i in range(nx_new)]
    ).T
    return result


print(
    f"\nInterpolating from {nx_orig}x{ny_orig} LGL grid to {interp_resolution}x{interp_resolution} uniform grid..."
)

U = interp_2d_barycentric(U_orig_2d, x_lgl, y_lgl, x_fine, y_fine)
V = interp_2d_barycentric(V_orig_2d, x_lgl, y_lgl, x_fine, y_fine)
P = interp_2d_barycentric(P_orig_2d, x_lgl, y_lgl, x_fine, y_fine)

# Compute velocity magnitude on fine grid
vel_mag = np.sqrt(U**2 + V**2)

# %%
# Create PyVista Grid
# -------------------

nx = ny = interp_resolution
n_points = nx * ny

# Create meshgrid coordinates
X, Y = np.meshgrid(x_fine, y_fine)

# Create structured grid
points = np.zeros((n_points, 3))
points[:, 0] = X.ravel()
points[:, 1] = Y.ravel()

grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [nx, ny, 1]

# Add scalar fields (flattened in row-major order) - use .copy() to avoid reference issues
grid.point_data["Velocity Magnitude"] = vel_mag.ravel().copy()
grid.point_data["U Velocity"] = U.ravel().copy()
grid.point_data["V Velocity"] = V.ravel().copy()
grid.point_data["Pressure"] = P.ravel().copy()

# Create velocity vectors for streamlines
velocity = np.zeros((n_points, 3))
velocity[:, 0] = U.ravel()
velocity[:, 1] = V.ravel()
grid.point_data["Velocity"] = velocity

print(f"PyVista grid created: {nx}x{ny} points (interpolated)")

# Verify data is correct
print("\nData verification:")
print(f"  U range: [{U.min():.4f}, {U.max():.4f}]")
print(f"  V range: [{V.min():.4f}, {V.max():.4f}]")
print(f"  vel_mag range: [{vel_mag.min():.4f}, {vel_mag.max():.4f}]")
print(f"  U at lid center (0.5, 1.0): {U[-1, nx // 2]:.4f}")
print(f"  V at lid center (0.5, 1.0): {V[-1, nx // 2]:.4f}")

# %%
# Plot 1: Velocity Magnitude with Streamlines
# -------------------------------------------

# Verify the data before plotting
vm_data = grid.point_data["Velocity Magnitude"]
print("\nPlot 1 - Using 'Velocity Magnitude' scalar:")
print(f"  Range: [{vm_data.min():.4f}, {vm_data.max():.4f}]")
print("  Should be sqrt(U^2+V^2), NOT same as U!")

plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])

# Add velocity magnitude as surface
plotter.add_mesh(
    grid,
    scalars="Velocity Magnitude",
    cmap="coolwarm",
    show_edges=False,
    lighting=False,
    scalar_bar_args={
        "title": "Velocity Magnitude",
        "title_font_size": 16,
        "label_font_size": 14,
        "shadow": True,
        "n_labels": 5,
        "italic": False,
        "fmt": "%.3f",
        "font_family": "arial",
        "vertical": True,
    },
)

# Generate streamlines
# Create seed points along the left boundary and bottom
n_seeds = 15
seed_points_left = np.zeros((n_seeds, 3))
seed_points_left[:, 0] = 0.02  # Slightly inside left boundary
seed_points_left[:, 1] = np.linspace(0.1, 0.9, n_seeds)

seed_points_bottom = np.zeros((n_seeds, 3))
seed_points_bottom[:, 0] = np.linspace(0.1, 0.9, n_seeds)
seed_points_bottom[:, 1] = 0.02  # Slightly inside bottom boundary

seed_points = np.vstack([seed_points_left, seed_points_bottom])
seeds = pv.PolyData(seed_points)

# Compute streamlines
try:
    streamlines = grid.streamlines_from_source(
        seeds,
        vectors="Velocity",
        integration_direction="both",
        max_length=50.0,
        initial_step_length=0.01,
        max_step_length=0.1,
    )

    if streamlines.n_points > 0:
        plotter.add_mesh(
            streamlines,
            color="white",
            line_width=1.5,
            opacity=0.8,
        )
        print(f"Added {streamlines.n_lines} streamlines")
except Exception as e:
    print(f"Streamline generation failed: {e}")

# Camera and labels
plotter.view_xy()
plotter.add_text(
    f"Lid-Driven Cavity Flow (Re={Re})\nSpectral Method - {nx}x{ny} Grid",
    position="upper_left",
    font_size=12,
    color="black",
)

# Add axes
plotter.add_axes(
    xlabel="x",
    ylabel="y",
    zlabel="",
    line_width=2,
    labels_off=False,
)

# Save figure
output_path = fig_dir / f"velocity_streamlines_Re{Re}.png"
plotter.screenshot(output_path, transparent_background=False, scale=2)
print(f"\nSaved: {output_path}")
plotter.close()

# %%
# Plot 2: Vorticity Field
# -----------------------

# Compute vorticity on the interpolated uniform grid using finite differences
# ω = ∂v/∂x - ∂u/∂y
dx = x_fine[1] - x_fine[0]
dy = y_fine[1] - y_fine[0]

# Central differences for interior, one-sided at boundaries
dv_dx = np.zeros_like(V)
du_dy = np.zeros_like(U)

# ∂v/∂x (central differences)
dv_dx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * dx)
dv_dx[:, 0] = (V[:, 1] - V[:, 0]) / dx
dv_dx[:, -1] = (V[:, -1] - V[:, -2]) / dx

# ∂u/∂y (central differences)
du_dy[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2 * dy)
du_dy[0, :] = (U[1, :] - U[0, :]) / dy
du_dy[-1, :] = (U[-1, :] - U[-2, :]) / dy

vorticity = dv_dx - du_dy
grid.point_data["Vorticity"] = vorticity.ravel()

plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])

# Symmetric colormap for vorticity
vort_max = np.abs(vorticity).max()

plotter.add_mesh(
    grid,
    scalars="Vorticity",
    cmap="RdBu_r",
    clim=[-vort_max, vort_max],
    show_edges=False,
    lighting=False,
    scalar_bar_args={
        "title": "Vorticity (ω)",
        "title_font_size": 16,
        "label_font_size": 14,
        "shadow": True,
        "n_labels": 5,
        "italic": False,
        "fmt": "%.2f",
        "font_family": "arial",
        "vertical": True,
    },
)

plotter.view_xy()
plotter.add_text(
    f"Vorticity Field (Re={Re})\nSpectral Method - {nx}x{ny} Grid",
    position="upper_left",
    font_size=12,
    color="black",
)

output_path = fig_dir / f"vorticity_Re{Re}.png"
plotter.screenshot(output_path, transparent_background=False, scale=2)
print(f"Saved: {output_path}")
plotter.close()

# %%
# Plot 3: Pressure Field
# ----------------------

plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])

plotter.add_mesh(
    grid,
    scalars="Pressure",
    cmap="coolwarm",
    show_edges=False,
    lighting=False,
    scalar_bar_args={
        "title": "Pressure",
        "title_font_size": 16,
        "label_font_size": 14,
        "shadow": True,
        "n_labels": 5,
        "italic": False,
        "fmt": "%.4f",
        "font_family": "arial",
        "vertical": True,
    },
)

plotter.view_xy()
plotter.add_text(
    f"Pressure Field (Re={Re})\nSpectral Method - {nx}x{ny} Grid",
    position="upper_left",
    font_size=12,
    color="black",
)

output_path = fig_dir / f"pressure_Re{Re}.png"
plotter.screenshot(output_path, transparent_background=False, scale=2)
print(f"Saved: {output_path}")
plotter.close()

# %%
# Plot 4: Velocity Components Side by Side
# ----------------------------------------

# Get actual data ranges for explicit clim
U_data = grid.point_data["U Velocity"]
V_data = grid.point_data["V Velocity"]
print("\nVelocity component data ranges:")
print(f"  U Velocity in grid: [{U_data.min():.4f}, {U_data.max():.4f}]")
print(f"  V Velocity in grid: [{V_data.min():.4f}, {V_data.max():.4f}]")

plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[2000, 900])

# U velocity - use coolwarm to show positive/negative
plotter.subplot(0, 0)
u_max = max(abs(U_data.min()), abs(U_data.max()))
plotter.add_mesh(
    grid,
    scalars="U Velocity",
    cmap="coolwarm",
    clim=[-u_max, u_max],  # symmetric around zero
    show_edges=False,
    lighting=False,
    scalar_bar_args={
        "title": "U Velocity",
        "title_font_size": 14,
        "label_font_size": 12,
        "n_labels": 5,
        "fmt": "%.3f",
        "vertical": True,
    },
)
plotter.view_xy()
plotter.add_text(
    f"U (horizontal)\nrange: [{U_data.min():.2f}, {U_data.max():.2f}]",
    position="upper_left",
    font_size=10,
    color="black",
)

# V velocity - use coolwarm with symmetric limits
plotter.subplot(0, 1)
v_max = max(abs(V_data.min()), abs(V_data.max()))
plotter.add_mesh(
    grid,
    scalars="V Velocity",
    cmap="coolwarm",
    clim=[-v_max, v_max],  # symmetric around zero
    show_edges=False,
    lighting=False,
    scalar_bar_args={
        "title": "V Velocity",
        "title_font_size": 14,
        "label_font_size": 12,
        "n_labels": 5,
        "fmt": "%.3f",
        "vertical": True,
    },
)
plotter.view_xy()
plotter.add_text(
    f"V (vertical)\nrange: [{V_data.min():.2f}, {V_data.max():.2f}]",
    position="upper_left",
    font_size=10,
    color="black",
)

output_path = fig_dir / f"velocity_components_Re{Re}.png"
plotter.screenshot(output_path, transparent_background=False, scale=2)
print(f"Saved: {output_path}")
plotter.close()

# %%
# Summary
# -------

print(f"\n{'=' * 50}")
print("Visualization complete!")
print(f"{'=' * 50}")
print(f"Data loaded from: {data_file}")
print(f"Output directory: {fig_dir}")
print("\nGenerated figures:")
print(f"  - velocity_streamlines_Re{Re}.png")
print(f"  - vorticity_Re{Re}.png")
print(f"  - pressure_Re{Re}.png")
print(f"  - velocity_components_Re{Re}.png")
