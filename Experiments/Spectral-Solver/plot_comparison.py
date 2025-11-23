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

# Load both solutions
plotter_leg = LDCPlotter(legendre_path)
plotter_cheb = LDCPlotter(chebyshev_path)

validator_leg = GhiaValidator(legendre_path, Re=Re, method_label='Legendre')
validator_cheb = GhiaValidator(chebyshev_path, Re=Re, method_label='Chebyshev')

print(f"Loaded solutions for Re={Re}:")
print(f"  Legendre:  {legendre_path.name}")
print(f"  Chebyshev: {chebyshev_path.name}")

# %%
# Ghia Validation Comparison
# ---------------------------
# Side-by-side Ghia benchmark comparison using clean DataFrame API

plot_validation(
    [validator_leg, validator_cheb],
    output_path=fig_dir / f"comparison_ghia_validation_{Re_str}.pdf"
)
print(f"  ✓ Ghia validation comparison saved")

# %%
# Convergence History Comparison
# -------------------------------
# Compare convergence behavior between Legendre and Chebyshev

plotter_leg.plot_convergence(output_path=fig_dir / f"legendre_convergence_{Re_str}.pdf")
print(f"  ✓ Legendre convergence saved")

plotter_cheb.plot_convergence(output_path=fig_dir / f"chebyshev_convergence_{Re_str}.pdf")
print(f"  ✓ Chebyshev convergence saved")

# %%
# Legendre Field Plots
# --------------------
# Velocity fields, pressure, and velocity magnitude with streamlines

plotter_leg.plot_velocity_fields(output_path=fig_dir / f"legendre_velocity_fields_{Re_str}.pdf")
print(f"  ✓ Legendre velocity fields saved")

plotter_leg.plot_pressure(output_path=fig_dir / f"legendre_pressure_{Re_str}.pdf")
print(f"  ✓ Legendre pressure saved")

plotter_leg.plot_velocity_magnitude(output_path=fig_dir / f"legendre_velocity_magnitude_{Re_str}.pdf")
print(f"  ✓ Legendre velocity magnitude saved")

# %%
# Chebyshev Field Plots
# ---------------------
# Velocity fields, pressure, and velocity magnitude with streamlines

plotter_cheb.plot_velocity_fields(output_path=fig_dir / f"chebyshev_velocity_fields_{Re_str}.pdf")
print(f"  ✓ Chebyshev velocity fields saved")

plotter_cheb.plot_pressure(output_path=fig_dir / f"chebyshev_pressure_{Re_str}.pdf")
print(f"  ✓ Chebyshev pressure saved")

plotter_cheb.plot_velocity_magnitude(output_path=fig_dir / f"chebyshev_velocity_magnitude_{Re_str}.pdf")
print(f"  ✓ Chebyshev velocity magnitude saved")

