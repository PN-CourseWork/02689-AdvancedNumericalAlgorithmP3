"""
Lid-Driven Cavity Flow Visualization (with FV vs Spectral comparison)
=====================================================================

This script visualizes the computed lid-driven cavity flow solution (FV)
and validates results vs Ghia. It also optionally interpolates a spectral
solution to the FV grid (or vice-versa) and computes L2 errors and simple
comparison plots.

Drop this file into `Experiments/01-FV-Solver/` and run from the project
root (so get_project_root() works). It detects spectral files placed in
`data/Spectral-Solver` and `data/Spectral-Solver/Chebyshev` based on the
naming convention in your repo.
"""

# %%
# Setup and Load Data
# -------------------
from pathlib import Path
import numpy as np

from utils import get_project_root, LDCPlotter, GhiaValidator, plot_validation

# Optional: SciPy for interpolation (needed for spectral/FV comparison)
try:
    from scipy.interpolate import BarycentricInterpolator
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed – interpolation-based comparison disabled.")

# matplotlib for optional plotting
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

# h5py for reading spectral/FV fields
import h5py

# Configuration (edit N or Re if needed)
Re = 100
N = 64  # FV grid size used to pick the FV file
Re_str = f"Re{int(Re)}"

project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
fig_dir = project_root / "figures" / "FV-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

# File path for FV solution
solution_path = data_dir / f"LDC_N{N}_{Re_str}.h5"
if not solution_path.exists():
    raise FileNotFoundError(f"FV solution not found: {solution_path}.\n"
                            f"Available files: {list(data_dir.glob('*.h5'))}")

# Load helpers
plotter = LDCPlotter(solution_path)
validator = GhiaValidator(solution_path, Re=Re, method_label="FV-SIMPLE")
print(f"Loaded FV solution: {solution_path.name}")

# %%
# Ghia Validation and basic plotting
plot_validation(validator, output_path=fig_dir / f"ghia_validation_{Re_str}.pdf")
print("  ✓ Ghia validation saved")
plotter.plot_convergence(output_path=fig_dir / f"convergence_{Re_str}.pdf")
print("  ✓ Convergence saved")
plotter.plot_fields(output_path=fig_dir / f"fields_{Re_str}.pdf")
print("  ✓ Fields saved")
plotter.plot_streamlines(output_path=fig_dir / f"streamlines_{Re_str}.pdf")
print("  ✓ Streamlines saved")

# =============================================================================
# Interpolation utilities
# =============================================================================

def try_dataset(f, keys):
    """Return first available dataset from keys in HDF5 file f."""
    for k in keys:
        if k in f:
            return np.asarray(f[k][:])
    raise KeyError(f"None of {keys} found in HDF5 file.")


def load_ldc_h5_fields(path: Path):
    """
    Robust loader for the HDF5 layout used in your FV and spectral files.

    It supports the block format:
      fields/block0_items  -> list of names (e.g. ['u','v','p','x','y'])
      fields/block0_values -> shape (npoints, nvars)  (here 4096 x 5)

    Returns:
      x (nx,), y (ny,), P (ny,nx), U (ny,nx), V (ny,nx)
    """
    import math

    with h5py.File(path, "r") as f:
        # --- Preferred block layout (observed in your FV files) ---
        if "fields" in f and "block0_items" in f["fields"] and "block0_values" in f["fields"]:
            raw_items = f["fields/block0_items"][:]
            # decode bytes -> str if necessary
            items = [ti.decode() if isinstance(ti, (bytes, bytearray)) else str(ti) for ti in raw_items]

            vals = np.asarray(f["fields/block0_values"][:])  # (npoints, nvars)
            npoints, nvars = vals.shape

            # infer square grid (npoints should be 64*64 = 4096)
            sq = int(math.isqrt(npoints))
            if sq * sq == npoints:
                nx = ny = sq
            else:
                raise ValueError(f"Cannot infer square grid from npoints={npoints}.")

            # build name->col index map (lowercased)
            colmap = {name.strip().lower(): i for i, name in enumerate(items)}

            # expected names in your file: 'u','v','p','x','y'
            def find_index(*cands):
                for c in cands:
                    if c in colmap:
                        return colmap[c]
                return None

            iu = find_index("u", "ux", "u_velocity")
            iv = find_index("v", "vy", "v_velocity", "uy")
            ip = find_index("p", "pressure", "pres")
            ix = find_index("x", "xc", "x_coords")
            iy = find_index("y", "yc", "y_coords")

            if None in (iu, iv, ip, ix, iy):
                raise KeyError(f"Could not find all required columns in block0_items (found {items}).")

            # extract flattened coords and fields
            x_flat = vals[:, ix]
            y_flat = vals[:, iy]

            # sort rows by (y, x) to get a structured grid and reshape
            order = np.lexsort((x_flat, y_flat))
            vals_sorted = vals[order, :]

            U = vals_sorted[:, iu].reshape((ny, nx))
            V = vals_sorted[:, iv].reshape((ny, nx))
            P = vals_sorted[:, ip].reshape((ny, nx))

            # unique x and y values (sorted)
            x = np.unique(vals_sorted[:, ix])
            y = np.unique(vals_sorted[:, iy])

            return x, y, P, U, V

        # --- Fallback: try conventional dataset names at top-level or groups ---
        # Try to locate datasets by common names
        x_candidates = ["x", "X", "xc", "xc_cell", "x_coords", "xx"]
        y_candidates = ["y", "Y", "yc", "yc_cell", "y_coords", "yy"]
        p_candidates = ["p", "P", "pressure", "pres"]
        u_candidates = ["u", "U", "ux", "u_velocity"]
        v_candidates = ["v", "V", "uy", "v_velocity"]

        x = try_dataset(f, x_candidates)
        y = try_dataset(f, y_candidates)
        P_raw = try_dataset(f, p_candidates)
        U_raw = try_dataset(f, u_candidates)
        V_raw = try_dataset(f, v_candidates)

    # ensure arrays and shape orientation
    P = np.asarray(P_raw)
    U = np.asarray(U_raw)
    V = np.asarray(V_raw)

    def fix_shape(A, nx=len(x), ny=len(y)):
        A = np.asarray(A)
        if A.shape == (ny, nx):
            return A
        if A.shape == (nx, ny):
            return A.T
        if A.size == nx * ny:
            return A.reshape((ny, nx))
        raise ValueError(f"Field has incompatible shape {A.shape} for grid sizes nx={nx}, ny={ny}")

    P = fix_shape(P)
    U = fix_shape(U)
    V = fix_shape(V)

    return x, y, P, U, V


# Barycentric interp helper
if SCIPY_AVAILABLE:
    from scipy.interpolate import BarycentricInterpolator
else:
    BarycentricInterpolator = None

def barycentric_interp_2d(x_unique, y_unique, field_2d, x_target, y_target):
    """Tensor-product barycentric interpolation from (x_unique,y_unique) to target grid."""
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not available; cannot perform barycentric interpolation.")

    x_unique = np.asarray(x_unique)
    y_unique = np.asarray(y_unique)
    field_2d = np.asarray(field_2d)

    # interpolate along x for each y
    temp = np.array([BarycentricInterpolator(x_unique, row)(x_target) for row in field_2d])
    # interpolate along y for each x
    Nx_t = len(x_target)
    Ny_t = len(y_target)
    result = np.empty((Ny_t, Nx_t))
    for i in range(Nx_t):
        col_interp = BarycentricInterpolator(y_unique, temp[:, i])
        result[:, i] = col_interp(y_target)
    X, Y = np.meshgrid(x_target, y_target)
    return X, Y, result


# =============================================================================
# FV vs Spectral comparison: find spectral file and compare
# =============================================================================

if SCIPY_AVAILABLE:
    # Search common spectral directories
    spectral_dirs = [project_root / "data" / "Spectral-Solver",
                     project_root / "data" / "Spectral-Solver" / "Chebyshev"]

    spectral_files = []
    for d in spectral_dirs:
        if d.exists():
            spectral_files.extend(sorted(d.glob(f"LDC_N*_Re{int(Re)}.h5")))

    print(f"Found spectral candidate files: {spectral_files}")

    # pick the spectral file with N closest to FV N (or choose the largest)
    def extract_N_from_name(p: Path):
        name = p.stem
        import re
        m = re.search(r"_N?(\d+)_Re", name)
        if m:
            return int(m.group(1))
        # try alternative patterns
        m = re.search(r"N(\d+)", name)
        if m:
            return int(m.group(1))
        return None

    best_spec = None
    if spectral_files:
        spec_candidates = [(abs((extract_N_from_name(p) or 0) - N), p) for p in spectral_files]
        spec_candidates = [t for t in spec_candidates if t[0] is not None]
        spec_candidates.sort(key=lambda t: (t[0], -extract_N_from_name(t[1]) if extract_N_from_name(t[1]) else 0))
        best_spec = spec_candidates[0][1]

    if best_spec is None:
        print("No spectral file found to compare against. Skipping comparison.")
    else:
        print(f"Selected spectral file for comparison: {best_spec}")

        # load FV fields
        x_fv, y_fv, P_fv, U_fv, V_fv = load_ldc_h5_fields(solution_path)
        # load spectral fields
        x_spec, y_spec, P_spec, U_spec, V_spec = load_ldc_h5_fields(best_spec)

        # Interpolate spectral -> FV grid
        x_target = x_fv
        y_target = y_fv
        print("Interpolating spectral fields onto FV grid...")
        X_fv, Y_fv, P_spec_interp = barycentric_interp_2d(x_spec, y_spec, P_spec, x_target, y_target)
        _, _, U_spec_interp = barycentric_interp_2d(x_spec, y_spec, U_spec, x_target, y_target)
        _, _, V_spec_interp = barycentric_interp_2d(x_spec, y_spec, V_spec, x_target, y_target)

        def l2_norm(a, b):
            return np.sqrt(np.mean((a - b) ** 2))

        err_P = l2_norm(P_fv, P_spec_interp)
        err_U = l2_norm(U_fv, U_spec_interp)
        err_V = l2_norm(V_fv, V_spec_interp)

        print("L2 errors (FV vs spectral interpolated to FV grid):")
        print(f"  ||P_FV - P_spec||_2 = {err_P:.4e}")
        print(f"  ||U_FV - U_spec||_2 = {err_U:.4e}")
        print(f"  ||V_FV - V_spec||_2 = {err_V:.4e}")

        # plotting
        if MPL_AVAILABLE:
            cmp_dir = fig_dir / "comparison"
            cmp_dir.mkdir(parents=True, exist_ok=True)

            def plot_field_comparison(field_fv, field_spec, title, fname):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                im0 = axes[0].pcolormesh(X_fv, Y_fv, field_fv, shading="auto")
                axes[0].set_title("FV")
                fig.colorbar(im0, ax=axes[0])

                im1 = axes[1].pcolormesh(X_fv, Y_fv, field_spec, shading="auto")
                axes[1].set_title("Spectral (interp)")
                fig.colorbar(im1, ax=axes[1])

                diff = field_fv - field_spec
                im2 = axes[2].pcolormesh(X_fv, Y_fv, diff, shading="auto")
                axes[2].set_title("Difference (FV - Spec)")
                fig.colorbar(im2, ax=axes[2])

                for ax in axes:
                    ax.set_aspect("equal")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")

                fig.suptitle(title)
                out_path = cmp_dir / fname
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
                print(f"  ✓ Saved comparison plot: {out_path}")

            plot_field_comparison(P_fv, P_spec_interp, "Pressure comparison", f"P_compare_{Re_str}.pdf")
            plot_field_comparison(U_fv, U_spec_interp, "u-velocity comparison", f"U_compare_{Re_str}.pdf")
            plot_field_comparison(V_fv, V_spec_interp, "v-velocity comparison", f"V_compare_{Re_str}.pdf")
        else:
            print("matplotlib not available — skipping saving comparison plots")

else:
    print("SciPy not available – skipping interpolation-based comparison.")