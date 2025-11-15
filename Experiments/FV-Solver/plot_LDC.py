#!/usr/bin/env python3
"""
Lid-Driven Cavity Flow Visualization
=====================================

This script visualizes the computed lid-driven cavity flow solution and validates
the results against the benchmark data from Ghia et al. (1982).

The visualization includes:
- Convergence history of the SIMPLE iteration
- Velocity vector fields (u and v components)
- Pressure field contours
- Comparison with Ghia benchmark profiles
"""

# %% Imports and Setup
# Import visualization and validation utilities
from utils import get_project_root, LDCPlotter, GhiaValidator

# Setup paths for data and figure output
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
fig_dir = project_root / "figures" / "FV-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

# %% Load Solution Data
# Initialize plotter with the computed HDF5 solution
plotter = LDCPlotter(data_dir / "LDC_Re100.h5")

print(f"Loaded solution from: {data_dir / 'LDC_Re100.h5'}")

# %% Plot Convergence History
# Visualize the residual decrease during SIMPLE iteration
plotter.plot_convergence(output_path=fig_dir / "LDC_Re100_convergence.pdf")
print(f"  ✓ Convergence plot saved")

# %% Plot Velocity Fields
# Generate velocity vector field visualization
plotter.plot_velocity_fields(output_path=fig_dir / "LDC_Re100_velocity.pdf")
print(f"  ✓ Velocity field plots saved")

# %% Plot Pressure Field
# Generate pressure contour visualization
plotter.plot_pressure(output_path=fig_dir / "LDC_Re100_pressure.pdf")
print(f"  ✓ Pressure field plot saved")

# %% Validate Against Ghia Benchmark
# Compare computed velocity profiles with Ghia et al. (1982) benchmark data
validator = GhiaValidator(h5_path=data_dir / "LDC_Re100.h5")
validator.plot_validation(output_path=fig_dir / "LDC_Re100_ghia_validation.pdf")
print(f"  ✓ Ghia validation plot saved")

print(f"\nAll figures saved to: {fig_dir}")
