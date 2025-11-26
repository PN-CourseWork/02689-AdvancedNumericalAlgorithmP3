"""
Legendre vs Chebyshev Basis Comparison Plots
=============================================

Compare the two spectral basis implementations side-by-side.
"""

# %%
# Setup
# -----
from utils import get_project_root, LDCPlotter
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

# Load both solutions
plotter_leg = LDCPlotter(legendre_path)
plotter_cheb = LDCPlotter(chebyshev_path)

print(f"Loaded solutions for Re={Re}:")
print(f"  Legendre:  {legendre_path.name}")
print(f"  Chebyshev: {chebyshev_path.name}")

# %%
# Ghia Validation Comparison
# ---------------------------
# Side-by-side Ghia benchmark comparison using clean DataFrame API
# NOTE: Skipping validation plots as Ghia benchmark data files are not available

# validator_leg = GhiaValidator(legendre_path, Re=Re, method_label='Legendre')
# validator_cheb = GhiaValidator(chebyshev_path, Re=Re, method_label='Chebyshev')
# plot_validation(
#     [validator_leg, validator_cheb],
#     output_path=fig_dir / f"comparison_ghia_validation_{Re_str}.pdf"
# )
# print(f"  ✓ Ghia validation comparison saved")
print(f"  ⊘ Skipping Ghia validation (benchmark data not available)")

# %%
# Convergence History Comparison
# -------------------------------
# Compare convergence behavior between Legendre and Chebyshev

fig_dir_leg = fig_dir / "Legendre"
fig_dir_leg.mkdir(parents=True, exist_ok=True)

plotter_leg.plot_convergence(output_path=fig_dir_leg / f"convergence_{Re_str}.pdf")
print(f"  ✓ Legendre convergence saved")

fig_dir_cheb = fig_dir / "Chebyshev"
fig_dir_cheb.mkdir(parents=True, exist_ok=True)

plotter_cheb.plot_convergence(output_path=fig_dir_cheb / f"convergence_{Re_str}.pdf")
print(f"  ✓ Chebyshev convergence saved")

# %%
# Legendre Field Plots
# --------------------
# Solution fields and streamlines

plotter_leg.plot_fields(output_path=fig_dir_leg / f"fields_{Re_str}.pdf")
print(f"  ✓ Legendre fields saved")

plotter_leg.plot_streamlines(output_path=fig_dir_leg / f"streamlines_{Re_str}.pdf")
print(f"  ✓ Legendre streamlines saved")

# %%
# Chebyshev Field Plots
# ---------------------
# Solution fields and streamlines

plotter_cheb.plot_fields(output_path=fig_dir_cheb / f"fields_{Re_str}.pdf")
print(f"  ✓ Chebyshev fields saved")

plotter_cheb.plot_streamlines(output_path=fig_dir_cheb / f"streamlines_{Re_str}.pdf")
print(f"  ✓ Chebyshev streamlines saved")

