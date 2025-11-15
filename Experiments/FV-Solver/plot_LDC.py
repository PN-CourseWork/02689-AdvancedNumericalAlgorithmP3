"""
Lid-Driven Cavity Flow Visualization
=====================================

This script visualizes the computed lid-driven cavity flow solution and validates
the results against the benchmark data from Ghia et al. (1982).
"""

# %%
# Setup and Load Data
# -------------------
# Import visualization utilities and load the computed solution from HDF5 file.

from utils import get_project_root, LDCPlotter, GhiaValidator

project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
fig_dir = project_root / "figures" / "FV-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

plotter = LDCPlotter(data_dir / "LDC_Re100.h5")
print(f"Loaded solution from: {data_dir / 'LDC_Re100.h5'}")

# %%
# Convergence History
# -------------------
# Visualize how the residual decreased during the SIMPLE iteration process.

plotter.plot_convergence(output_path=fig_dir / "LDC_Re100_convergence.pdf")
print(f"  ✓ Convergence plot saved")

# %%
# Velocity Fields
# ---------------
# Generate velocity vector field visualizations for the u and v components.

plotter.plot_velocity_fields(output_path=fig_dir / "LDC_Re100_velocity.pdf")
print(f"  ✓ Velocity field plots saved")

# %%
# Pressure Field
# --------------
# Generate pressure contour visualization.

plotter.plot_pressure(output_path=fig_dir / "LDC_Re100_pressure.pdf")
print(f"  ✓ Pressure field plot saved")

# %%
# Ghia Benchmark Validation
# --------------------------
# Compare computed velocity profiles with the Ghia et al. (1982) benchmark data.

validator = GhiaValidator(h5_path=data_dir / "LDC_Re100.h5")
validator.plot_validation(output_path=fig_dir / "LDC_Re100_ghia_validation.pdf")
print(f"  ✓ Ghia validation plot saved")

print(f"\nAll figures saved to: {fig_dir}")
