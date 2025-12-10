"""
PyVista-based Field Visualization for LDC.

Generates high-quality 2D field plots using PyVista with the ParaView theme.
Loads solution directly from VTS files.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv

log = logging.getLogger(__name__)

# Enable off-screen rendering
os.environ["PYVISTA_OFF_SCREEN"] = "true"

# Use ParaView theme
pv.set_plot_theme("paraview")


# Window size for high DPI output (2400x2400 = 2x the previous 1200x1200)
WINDOW_SIZE = [2400, 2400]

# Common scalar bar arguments with black text and larger font
SCALAR_BAR_ARGS = {
    "vertical": False,
    "position_x": 0.25,
    "position_y": 0.02,  # Lower position for more padding
    "width": 0.5,
    "height": 0.04,
    "title_font_size": 44,  # Scaled up for higher resolution
    "label_font_size": 32,  # Scaled up for higher resolution
    "color": "black",
    "fmt": "%.2f",
    "n_labels": 5,
    "italic": True,  # Italic for more LaTeX-like appearance
    "font_family": "times",  # Times font for LaTeX-like appearance
}


def _setup_camera(plotter: pv.Plotter) -> None:
    """Set up proper 2D orthographic view after mesh is added."""
    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.reset_camera_clipping_range()
    plotter.reset_camera(bounds=plotter.bounds)


_STREAMLINES_SCRIPT = '''
import os
import sys
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import numpy as np
import pyvista as pv
pv.set_plot_theme("paraview")

vts_path = sys.argv[1]
output_path = sys.argv[2]
separating_distance = float(sys.argv[3])

WINDOW_SIZE = [2400, 2400]
SCALAR_BAR_ARGS = {
    "vertical": False,
    "position_x": 0.25,
    "position_y": 0.02,
    "width": 0.5,
    "height": 0.04,
    "title_font_size": 44,
    "label_font_size": 32,
    "color": "black",
    "fmt": "%.2f",
    "n_labels": 5,
    "italic": True,
    "font_family": "times",
}

grid = pv.read(vts_path)
plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)

plotter.add_mesh(
    grid,
    scalars="velocity_magnitude",
    cmap="coolwarm",
    show_edges=False,
    scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Velocity magnitude, |u|\\n"},
)

bounds = grid.bounds
resolution = 256
image_data = pv.ImageData(
    dimensions=(resolution, resolution, 1),
    spacing=((bounds[1] - bounds[0]) / (resolution - 1),
             (bounds[3] - bounds[2]) / (resolution - 1),
             1.0),
    origin=(bounds[0], bounds[2], 0.0),
)

sampled = image_data.sample(grid)

streamlines = sampled.streamlines_evenly_spaced_2D(
    vectors="velocity",
    start_position=(0.3, 0.7, 0.0),
    separating_distance=separating_distance,
    separating_distance_ratio=0.5,
    step_length=0.5,
    max_steps=500,  # Reduced from 800 to prevent runaway
    terminal_speed=1e-5,  # Stop integration when velocity gets very low
    closed_loop_maximum_distance=1.0,  # Help detect closed loops earlier
    loop_angle=30.0,  # More aggressive loop detection
)

if streamlines.n_points > 0:
    plotter.add_mesh(
        streamlines,
        color="white",
        line_width=2.0,
        opacity=0.2,
    )

plotter.enable_parallel_projection()
plotter.view_xy()
plotter.reset_camera_clipping_range()
plotter.reset_camera(bounds=plotter.bounds)

plotter.screenshot(output_path, transparent_background=True)
plotter.close()
'''


def _generate_streamlines_safe(vts_path: Path, output_path: Path, timeout: int = 60) -> bool:
    """Generate streamlines plot in a subprocess to isolate VTK crashes.

    Tries dense streamlines first, falls back to sparser if it crashes.
    """
    vts_str = str(vts_path)
    out_str = str(output_path)

    # Try dense streamlines first (separating_distance=2.5)
    for sep_dist in [2.5, 4.0, 8.0]:
        try:
            # Use start_new_session=True to detach subprocess from terminal,
            # preventing macOS crash reporter dialogs from appearing
            result = subprocess.run(
                [sys.executable, "-c", _STREAMLINES_SCRIPT, vts_str, out_str, str(sep_dist)],
                timeout=timeout,
                capture_output=True,
                text=True,
                start_new_session=True,
            )
            if result.returncode == 0:
                log.info(f"Streamlines generated with separating_distance={sep_dist}")
                return True
            else:
                log.warning(f"Streamlines failed with separating_distance={sep_dist}: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log.warning(f"Streamlines timed out with separating_distance={sep_dist}")
        except Exception as e:
            log.warning(f"Streamlines error with separating_distance={sep_dist}: {e}")

    log.error("All streamline attempts failed")
    return False


def plot_u_velocity(
    grid: pv.StructuredGrid,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Generate u-velocity contour plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.add_mesh(
        grid,
        scalars="u",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Horizontal velocity, u\n"},
    )
    _setup_camera(plotter)

    output_path = output_dir / f"u{suffix}.png"
    plotter.screenshot(output_path, transparent_background=True)
    plotter.close()

    log.info(f"Saved: {output_path}")
    return output_path


def plot_v_velocity(
    grid: pv.StructuredGrid,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Generate v-velocity contour plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.add_mesh(
        grid,
        scalars="v",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Vertical velocity, v\n"},
    )
    _setup_camera(plotter)

    output_path = output_dir / f"v{suffix}.png"
    plotter.screenshot(output_path, transparent_background=True)
    plotter.close()

    log.info(f"Saved: {output_path}")
    return output_path


def plot_velocity_magnitude(
    grid: pv.StructuredGrid,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Generate velocity magnitude contour plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.add_mesh(
        grid,
        scalars="velocity_magnitude",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Velocity magnitude, |u|\n"},
    )
    _setup_camera(plotter)

    output_path = output_dir / f"vel-mag{suffix}.png"
    plotter.screenshot(output_path, transparent_background=True)
    plotter.close()

    log.info(f"Saved: {output_path}")
    return output_path


def plot_pressure(
    grid: pv.StructuredGrid,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Generate pressure contour plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.add_mesh(
        grid,
        scalars="pressure",
        cmap="inferno",
        show_edges=False,
        scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Pressure, p\n"},
    )
    _setup_camera(plotter)

    output_path = output_dir / f"pressure{suffix}.png"
    plotter.screenshot(output_path, transparent_background=True)
    plotter.close()

    log.info(f"Saved: {output_path}")
    return output_path


def plot_streamlines(
    grid: pv.StructuredGrid,
    output_dir: Path,
    n_seeds: int = 8,
    suffix: str = "",
) -> Path:
    """Generate streamlines plot with velocity magnitude background.

    Parameters
    ----------
    grid : pv.StructuredGrid
        Solution grid with velocity vectors
    output_dir : Path
        Output directory
    n_seeds : int
        Number of seed points per dimension (reduced for less density)
    suffix : str
        Filename suffix

    Returns
    -------
    Path
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create seed points on a grid (avoiding exact boundaries)
    seed_x = np.linspace(0.1, 0.9, n_seeds)
    seed_y = np.linspace(0.1, 0.9, n_seeds)
    seed_X, seed_Y = np.meshgrid(seed_x, seed_y)
    seed_points = np.column_stack([
        seed_X.ravel(),
        seed_Y.ravel(),
        np.zeros(n_seeds * n_seeds)
    ])
    seeds = pv.PolyData(seed_points)

    # Compute streamlines
    streamlines = grid.streamlines_from_source(
        seeds,
        vectors="velocity",
        integration_direction="both",
        initial_step_length=0.01,
        max_step_length=0.1,
    )

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)

    # Add velocity magnitude as background
    plotter.add_mesh(
        grid,
        scalars="velocity_magnitude",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Velocity magnitude, |u|\n"},
    )

    # Add streamlines
    if streamlines.n_points > 0:
        plotter.add_mesh(
            streamlines,
            color="white",
            line_width=3.0,  # Scaled for higher resolution
            opacity=0.8,
        )

    _setup_camera(plotter)

    output_path = output_dir / f"streamlines{suffix}.png"
    plotter.screenshot(output_path, transparent_background=True)
    plotter.close()

    log.info(f"Saved: {output_path}")
    return output_path


def plot_streamlines_evenly_spaced(
    grid: pv.StructuredGrid,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Generate evenly-spaced streamlines using line sources.

    Uses line sources at multiple positions to create evenly distributed
    streamlines across the domain, similar to PyVista's 2D streamlines example.

    Parameters
    ----------
    grid : pv.StructuredGrid
        Solution grid with velocity vectors
    output_dir : Path
        Output directory
    suffix : str
        Filename suffix

    Returns
    -------
    Path
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use line sources for evenly spaced seeds (like PyVista example)
    # Vertical line at left side of domain
    streamlines_left = grid.streamlines(
        pointa=(0.05, 0.05, 0),
        pointb=(0.05, 0.95, 0),
        n_points=12,
        vectors="velocity",
        integration_direction="forward",
        initial_step_length=0.01,
        max_step_length=0.1,
        max_steps=500,
    )

    # Horizontal line at top of domain
    streamlines_top = grid.streamlines(
        pointa=(0.05, 0.95, 0),
        pointb=(0.95, 0.95, 0),
        n_points=12,
        vectors="velocity",
        integration_direction="forward",
        initial_step_length=0.01,
        max_step_length=0.1,
        max_steps=500,
    )

    # Vertical line at right side
    streamlines_right = grid.streamlines(
        pointa=(0.95, 0.95, 0),
        pointb=(0.95, 0.05, 0),
        n_points=12,
        vectors="velocity",
        integration_direction="forward",
        initial_step_length=0.01,
        max_step_length=0.1,
        max_steps=500,
    )

    # Horizontal line at bottom
    streamlines_bottom = grid.streamlines(
        pointa=(0.95, 0.05, 0),
        pointb=(0.05, 0.05, 0),
        n_points=12,
        vectors="velocity",
        integration_direction="forward",
        initial_step_length=0.01,
        max_step_length=0.1,
        max_steps=500,
    )

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)

    # Add velocity magnitude as background
    plotter.add_mesh(
        grid,
        scalars="velocity_magnitude",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Velocity magnitude, |u|\n"},
    )

    # Add all streamlines
    for streamlines in [streamlines_left, streamlines_top, streamlines_right, streamlines_bottom]:
        if streamlines.n_points > 0:
            plotter.add_mesh(
                streamlines,
                color="white",
                line_width=3.0,  # Scaled for higher resolution
                opacity=0.9,
            )

    _setup_camera(plotter)

    output_path = output_dir / f"streamlines{suffix}.png"
    plotter.screenshot(output_path, transparent_background=True)
    plotter.close()

    log.info(f"Saved: {output_path}")
    return output_path


def plot_streamlines_evenly_spaced_2D(
    grid: pv.StructuredGrid,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Generate evenly-spaced streamlines using VTK's 2D algorithm.

    Uses streamlines_evenly_spaced_2D for publication-quality streamlines.
    This algorithm produces streamlines that are approximately evenly spaced.

    Parameters
    ----------
    grid : pv.StructuredGrid
        Solution grid with velocity vectors
    output_dir : Path
        Output directory
    suffix : str
        Filename suffix

    Returns
    -------
    Path
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)

    # Add velocity magnitude as background
    plotter.add_mesh(
        grid,
        scalars="velocity_magnitude",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Velocity magnitude, |u|\n"},
    )

    # Use evenly spaced streamlines 2D algorithm
    # First resample to ImageData for better algorithm performance
    try:
        # Create uniform ImageData grid with higher resolution for smoother streamlines
        bounds = grid.bounds
        resolution = 256  # Balance between smoothness and stability
        image_data = pv.ImageData(
            dimensions=(resolution, resolution, 1),
            spacing=((bounds[1] - bounds[0]) / (resolution - 1),
                     (bounds[3] - bounds[2]) / (resolution - 1),
                     1.0),
            origin=(bounds[0], bounds[2], 0.0),
        )

        # Sample the velocity field onto the uniform grid
        sampled = image_data.sample(grid)

        streamlines = sampled.streamlines_evenly_spaced_2D(
            vectors="velocity",
            start_position=(0.3, 0.7, 0.0),  # Near primary vortex
            separating_distance=8.0,  # Conservative for stability with non-uniform grids
            separating_distance_ratio=0.5,
            step_length=0.5,  # Step size in cell units
            max_steps=800,
        )

        if streamlines.n_points > 0:
            plotter.add_mesh(
                streamlines,
                color="white",
                line_width=2.0,  # Scaled for higher resolution
                opacity=0.2,  # Subtle but visible
            )
            log.info(f"Generated {streamlines.n_lines} evenly-spaced 2D streamlines")
    except Exception as e:
        log.warning(f"streamlines_evenly_spaced_2D failed: {e}")

    _setup_camera(plotter)

    output_path = output_dir / f"streamlines{suffix}.png"
    plotter.screenshot(output_path, transparent_background=True)
    plotter.close()

    log.info(f"Saved: {output_path}")
    return output_path


def generate_pyvista_field_plots(
    vts_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate PyVista-based field plots from a VTS solution file.

    Generates clean artifact names: u.png, v.png, pressure.png, vel-mag.png, streamlines.png

    Parameters
    ----------
    vts_path : Path
        Path to solution.vts file
    output_dir : Path
        Output directory

    Returns
    -------
    dict[str, Path]
        Dictionary mapping plot names to their paths
    """
    vts_path = Path(vts_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load solution
    grid = pv.read(str(vts_path))

    paths = {}

    log.info("Generating PyVista field plots...")

    log.info("  u velocity...")
    paths["u"] = plot_u_velocity(grid, output_dir, suffix="")

    log.info("  v velocity...")
    paths["v"] = plot_v_velocity(grid, output_dir, suffix="")

    log.info("  velocity magnitude...")
    paths["vel-mag"] = plot_velocity_magnitude(grid, output_dir, suffix="")

    log.info("  pressure...")
    paths["pressure"] = plot_pressure(grid, output_dir, suffix="")

    log.info("  streamlines (2D evenly spaced)...")
    streamlines_path = output_dir / "streamlines.png"
    if _generate_streamlines_safe(vts_path, streamlines_path):
        paths["streamlines"] = streamlines_path

    return paths
