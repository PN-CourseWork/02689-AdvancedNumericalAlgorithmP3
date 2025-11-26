"""
Legendre vs Chebyshev Basis Comparison Plots
=============================================

Compare the two spectral basis implementations side-by-side.
"""

# %%
# Setup
# -----
from utils import get_project_root, LDCPlotter, GhiaValidator, plot_validation
from pathlib import Path

# Configuration
Re = 100
N = 16  # Grid size (number of nodes)
Re_str = f"Re{int(Re)}"

project_root = get_project_root()
data_dir = project_root / "data" / "Spectral-Solver"
fig_dir = project_root / "figures" / "Spectral-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

# File paths
legendre_path = data_dir / "Legendre" / f"LDC_N{N}_{Re_str}.h5"
chebyshev_path = data_dir / "Chebyshev" / f"LDC_N{N}_{Re_str}.h5"

# Validate paths exist
if not legendre_path.exists():
    raise FileNotFoundError(f"Legendre solution not found: {legendre_path}")
if not chebyshev_path.exists():
    raise FileNotFoundError(f"Chebyshev solution not found: {chebyshev_path}")

# Load Chebyshev solution (Legendre diverged with current settings)
plotter_cheb = LDCPlotter(chebyshev_path)
validator_cheb = GhiaValidator(chebyshev_path, Re=Re, method_label='Chebyshev')

print(f"Loaded solutions for Re={Re}:")
print(f"  Chebyshev: {chebyshev_path.name}")

# %%
# Ghia Validation
# ---------------
# Ghia benchmark validation for Chebyshev spectral solver

plot_validation(
    validator_cheb,
    output_path=fig_dir / f"comparison_ghia_validation_{Re_str}.pdf"
)
print(f"  ✓ Ghia validation saved")

# %%
# Convergence History
# -------------------
# Chebyshev convergence behavior

fig_dir_cheb = fig_dir / "Chebyshev"
fig_dir_cheb.mkdir(parents=True, exist_ok=True)

plotter_cheb.plot_convergence(output_path=fig_dir_cheb / f"convergence_{Re_str}.pdf")
print(f"  ✓ Chebyshev convergence saved")

# %%
# Chebyshev Field Plots
# ---------------------
# Solution fields and streamlines

plotter_cheb.plot_fields(output_path=fig_dir_cheb / f"fields_{Re_str}.pdf")
print(f"  ✓ Chebyshev fields saved")

plotter_cheb.plot_streamlines(output_path=fig_dir_cheb / f"streamlines_{Re_str}.pdf")
print(f"  ✓ Chebyshev streamlines saved")

