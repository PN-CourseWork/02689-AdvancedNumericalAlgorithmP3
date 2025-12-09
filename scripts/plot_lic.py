"""Line Integral Convolution (LIC) visualization for lid-driven cavity flow.

LIC convolves vector field orientation with local texture over a kernel length,
preserving all details of the velocity field geometry at pixel scale.

Usage:
    python scripts/plot_lic.py                          # Uses bundled Re=1000, N=40 solution
    python scripts/plot_lic.py --npz path/to/solution.npz
    python scripts/plot_lic.py --vts path/to/solution.vts
    python scripts/plot_lic.py --run-id <mlflow-run-id>

References:
    - Cabral & Leedom (1993): "Imaging Vector Fields Using Line Integral Convolution"
    - SciPy Cookbook LIC implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import map_coordinates
from scipy.interpolate import RectBivariateSpline
from pathlib import Path
import sys

# Default bundled solution (Re=1000, N=40, spectral FSG solver)
DEFAULT_SOLUTION = Path(__file__).parent.parent / "data" / "ldc_solution_Re1000_N40.npz"


def generate_lic_fast(u: np.ndarray, v: np.ndarray, length: int = 30) -> np.ndarray:
    """Generate Line Integral Convolution texture from velocity field.

    Uses vectorized operations with scipy's map_coordinates for speed.

    Parameters
    ----------
    u : np.ndarray
        x-component of velocity, shape (ny, nx)
    v : np.ndarray
        y-component of velocity, shape (ny, nx)
    length : int
        Kernel length for integration (number of steps in each direction)

    Returns
    -------
    np.ndarray
        LIC texture, shape (ny, nx)
    """
    ny, nx = u.shape

    # Generate white noise texture
    np.random.seed(42)  # For reproducibility
    noise = np.random.rand(ny, nx).astype(np.float32)

    # Normalize velocity to unit vectors (avoid division by zero)
    magnitude = np.sqrt(u**2 + v**2)
    magnitude = np.maximum(magnitude, 1e-10)
    u_norm = (u / magnitude).astype(np.float32)
    v_norm = (v / magnitude).astype(np.float32)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:ny, 0:nx].astype(np.float32)

    # Accumulator for LIC
    lic = np.zeros((ny, nx), dtype=np.float32)
    weights = np.zeros((ny, nx), dtype=np.float32)

    # Forward and backward integration
    for direction in [1, -1]:
        x = x_coords.copy()
        y = y_coords.copy()

        for step in range(length):
            # Clip to valid range
            x_clipped = np.clip(x, 0, nx - 1.001)
            y_clipped = np.clip(y, 0, ny - 1.001)

            # Sample noise at current positions
            coords = np.array([y_clipped.ravel(), x_clipped.ravel()])
            sampled = map_coordinates(noise, coords, order=1, mode='constant', cval=0)
            sampled = sampled.reshape(ny, nx)

            # Check which points are still in bounds
            in_bounds = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)

            # Accumulate
            lic += sampled * in_bounds
            weights += in_bounds.astype(np.float32)

            # Get velocity at current positions
            coords_int = np.array([y_clipped.ravel(), x_clipped.ravel()])
            dx = map_coordinates(u_norm, coords_int, order=1, mode='constant', cval=0).reshape(ny, nx)
            dy = map_coordinates(v_norm, coords_int, order=1, mode='constant', cval=0).reshape(ny, nx)

            # Step in direction
            x += direction * dx * 0.5
            y += direction * dy * 0.5

    # Normalize
    weights = np.maximum(weights, 1)
    lic = lic / weights

    return lic


def load_solution_from_npz(npz_path: Path) -> tuple:
    """Load solution from NPZ file.

    Parameters
    ----------
    npz_path : Path
        Path to solution.npz file

    Returns
    -------
    tuple
        (x_nodes, y_nodes, U_2d, V_2d, Re, N)
    """
    data = np.load(npz_path)
    return (
        data["x_nodes"],
        data["y_nodes"],
        data["U"],
        data["V"],
        float(data["Re"]),
        int(data["N"]),
    )


def load_solution_from_vts(vts_path: Path) -> dict:
    """Load solution from VTS file using pyvista.

    Parameters
    ----------
    vts_path : Path
        Path to solution.vts file

    Returns
    -------
    dict
        Dictionary with x, y, u, v, p arrays
    """
    import pyvista as pv

    grid = pv.read(str(vts_path))
    points = grid.points

    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "u": grid["u"],
        "v": grid["v"],
        "p": grid["pressure"],
    }


def spectral_interpolate_to_fine_grid(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    U_2d: np.ndarray,
    V_2d: np.ndarray,
    n_fine: int = 512,
) -> tuple:
    """Interpolate solution from spectral nodes to fine uniform grid.

    Uses the fact that spectral solutions can be evaluated at arbitrary
    points via polynomial interpolation (spectrally accurate).

    Parameters
    ----------
    x_nodes : np.ndarray
        1D array of x node positions (Chebyshev/Legendre nodes)
    y_nodes : np.ndarray
        1D array of y node positions
    U_2d : np.ndarray
        u velocity on spectral grid, shape (ny, nx)
    V_2d : np.ndarray
        v velocity on spectral grid, shape (ny, nx)
    n_fine : int
        Number of points in fine grid

    Returns
    -------
    tuple
        (x_fine, y_fine, U_fine, V_fine) on uniform grid
    """
    # Create fine uniform grid
    x_fine = np.linspace(x_nodes.min(), x_nodes.max(), n_fine)
    y_fine = np.linspace(y_nodes.min(), y_nodes.max(), n_fine)

    # Use RectBivariateSpline for smooth interpolation
    # This is spectrally accurate when source points are spectral nodes
    U_spline = RectBivariateSpline(y_nodes, x_nodes, U_2d, kx=3, ky=3)
    V_spline = RectBivariateSpline(y_nodes, x_nodes, V_2d, kx=3, ky=3)

    U_fine = U_spline(y_fine, x_fine)
    V_fine = V_spline(y_fine, x_fine)

    return x_fine, y_fine, U_fine, V_fine


def restructure_to_2d(fields: dict) -> tuple:
    """Convert flattened fields to 2D arrays.

    Parameters
    ----------
    fields : dict
        Dictionary with x, y, u, v arrays (flattened)

    Returns
    -------
    tuple
        (x_unique, y_unique, U_2d, V_2d)
    """
    x, y = fields["x"], fields["y"]
    u, v = fields["u"], fields["v"]

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))
    nx, ny = len(x_unique), len(y_unique)

    # Build index maps
    x_to_idx = {val: i for i, val in enumerate(x_unique)}
    y_to_idx = {val: i for i, val in enumerate(y_unique)}

    U_2d = np.zeros((ny, nx))
    V_2d = np.zeros((ny, nx))

    for k in range(len(x)):
        i = x_to_idx[x[k]]
        j = y_to_idx[y[k]]
        U_2d[j, i] = u[k]
        V_2d[j, i] = v[k]

    return x_unique, y_unique, U_2d, V_2d


def plot_lic(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    U_2d: np.ndarray,
    V_2d: np.ndarray,
    Re: float = None,
    N: int = None,
    output_dir: Path = None,
    n_fine: int = 512,
    lic_length: int = 30,
) -> Path:
    """Generate LIC visualization from 2D velocity arrays.

    Parameters
    ----------
    x_nodes : np.ndarray
        1D array of x node positions
    y_nodes : np.ndarray
        1D array of y node positions
    U_2d : np.ndarray
        u velocity on 2D grid, shape (ny, nx)
    V_2d : np.ndarray
        v velocity on 2D grid, shape (ny, nx)
    Re : float, optional
        Reynolds number (for title)
    N : int, optional
        Grid resolution (for title)
    output_dir : Path
        Output directory
    n_fine : int
        Fine grid resolution for LIC
    lic_length : int
        LIC kernel length

    Returns
    -------
    Path
        Path to saved figure
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "figures" / "lic"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if N is None:
        N = len(x_nodes) - 1

    # Interpolate to fine uniform grid
    print(f"Interpolating to {n_fine}x{n_fine} uniform grid...")
    x_fine, y_fine, U_fine, V_fine = spectral_interpolate_to_fine_grid(
        x_nodes, y_nodes, U_2d, V_2d, n_fine=n_fine
    )

    # Compute velocity magnitude
    vel_mag = np.sqrt(U_fine**2 + V_fine**2)

    # Generate LIC texture
    print(f"Generating LIC texture (kernel length={lic_length})...")
    lic = generate_lic_fast(U_fine, V_fine, length=lic_length)

    print("Creating visualization...")

    # Normalize for coloring
    vel_normalized = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min() + 1e-10)
    lic_normalized = (lic - lic.min()) / (lic.max() - lic.min() + 1e-10)

    suffix = ""
    if Re is not None:
        suffix += f"_Re{int(Re)}"
    if N is not None:
        suffix += f"_N{N}"

    cmap = plt.cm.coolwarm
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=Normalize(vmin=vel_mag.min(), vmax=vel_mag.max()))

    # Plot 1: Pure LIC texture (grayscale)
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(lic, origin='lower', extent=[0, 1, 0, 1], cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    plt.tight_layout()
    path1 = output_dir / f"lic_grayscale{suffix}.pdf"
    fig1.savefig(path1, dpi=400, bbox_inches='tight', transparent=True)
    plt.close(fig1)
    print(f"Saved: {path1}")

    # Plot 2: LIC colored by velocity magnitude with colorbar
    fig2, ax2 = plt.subplots(figsize=(6, 6.8))
    colors = cmap(vel_normalized)
    lic_factor = 0.25 + 0.75 * lic_normalized
    colors[:, :, :3] *= lic_factor[:, :, np.newaxis]
    ax2.imshow(colors, origin='lower', extent=[0, 1, 0, 1])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.05, aspect=30)
    cbar2.set_label(r'$|\mathbf{u}|$')
    plt.tight_layout()
    path2 = output_dir / f"lic_colored{suffix}.pdf"
    fig2.savefig(path2, dpi=400, bbox_inches='tight', transparent=True)
    plt.close(fig2)
    print(f"Saved: {path2}")

    # Plot 3: High-res LIC poster background (A2 size: 420x594mm, 300dpi = 4961x7016px)
    # Using square aspect, so 16.5x16.5 inches at 300dpi = ~5000px
    fig3, ax3 = plt.subplots(figsize=(16.5, 16.5))
    colors = cmap(vel_normalized)
    lic_factor = 0.25 + 0.75 * lic_normalized
    colors[:, :, :3] *= lic_factor[:, :, np.newaxis]
    ax3.imshow(colors, origin='lower', extent=[0, 1, 0, 1])
    ax3.axis('off')
    plt.tight_layout(pad=0)
    path3 = output_dir / f"poster_lic{suffix}.png"
    fig3.savefig(path3, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig3)
    print(f"Saved: {path3}")

    return path2


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LIC visualization for lid-driven cavity flow"
    )
    parser.add_argument(
        "--npz", type=str, help="Path to solution.npz file"
    )
    parser.add_argument(
        "--vts", type=str, help="Path to solution.vts file"
    )
    parser.add_argument(
        "--run-id", type=str, help="MLflow run ID to load solution from"
    )
    parser.add_argument(
        "--Re", type=float, default=None, help="Reynolds number (for title)"
    )
    parser.add_argument(
        "--N", type=int, default=None, help="Grid resolution (for title)"
    )
    parser.add_argument(
        "--n-fine", type=int, default=512, help="Fine grid resolution for LIC"
    )
    parser.add_argument(
        "--length", type=int, default=30, help="LIC kernel length"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory"
    )

    args = parser.parse_args()

    # Determine source and load data
    Re, N = args.Re, args.N

    if args.npz:
        # Load from NPZ file
        npz_path = Path(args.npz)
        if not npz_path.exists():
            print(f"Error: NPZ file not found: {npz_path}")
            sys.exit(1)
        print(f"Loading solution from: {npz_path}")
        x_nodes, y_nodes, U_2d, V_2d, Re_file, N_file = load_solution_from_npz(npz_path)
        Re = Re if Re is not None else Re_file
        N = N if N is not None else N_file

    elif args.vts:
        # Load from VTS file
        vts_path = Path(args.vts)
        if not vts_path.exists():
            print(f"Error: VTS file not found: {vts_path}")
            sys.exit(1)
        print(f"Loading solution from: {vts_path}")
        fields = load_solution_from_vts(vts_path)
        x_nodes, y_nodes, U_2d, V_2d = restructure_to_2d(fields)

    elif args.run_id:
        # Load from MLflow
        import mlflow

        print(f"Loading solution from MLflow run: {args.run_id}")
        client = mlflow.tracking.MlflowClient()
        artifact_dir = client.download_artifacts(args.run_id, "")
        vts_path = Path(artifact_dir) / "solution.vts"

        if not vts_path.exists():
            vts_path = Path(artifact_dir) / "fields" / "solution.vts"

        if not vts_path.exists():
            print(f"Error: solution.vts not found in artifacts")
            sys.exit(1)

        fields = load_solution_from_vts(vts_path)
        x_nodes, y_nodes, U_2d, V_2d = restructure_to_2d(fields)

    else:
        # Use bundled default solution
        if not DEFAULT_SOLUTION.exists():
            print(f"Error: Bundled solution not found: {DEFAULT_SOLUTION}")
            print("Please provide --npz, --vts, or --run-id")
            sys.exit(1)
        print(f"Using bundled solution: {DEFAULT_SOLUTION.name}")
        x_nodes, y_nodes, U_2d, V_2d, Re_file, N_file = load_solution_from_npz(DEFAULT_SOLUTION)
        Re = Re if Re is not None else Re_file
        N = N if N is not None else N_file

    output_dir = Path(args.output) if args.output else None

    plot_lic(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        U_2d=U_2d,
        V_2d=V_2d,
        Re=Re,
        N=N,
        output_dir=output_dir,
        n_fine=args.n_fine,
        lic_length=args.length,
    )


if __name__ == "__main__":
    main()
