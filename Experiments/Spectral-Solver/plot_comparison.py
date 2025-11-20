"""
Legendre vs Chebyshev Basis Comparison Plots
=============================================

Compare the two spectral basis implementations side-by-side.

Usage:
    python plot_comparison.py [--Re RE] [--legendre PATH] [--chebyshev PATH]
"""

#TODO: Make this less hacky

# %%
# Setup
# -----
from utils import get_project_root, LDCPlotter, GhiaValidator
from utils.plotting import plt
from scipy.interpolate import interp1d
import numpy as np
import argparse
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare Legendre and Chebyshev spectral methods')
parser.add_argument('--Re', type=float, default=100, help='Reynolds number (default: 100)')
parser.add_argument('--legendre', type=str, help='Path to Legendre HDF5 file')
parser.add_argument('--chebyshev', type=str, help='Path to Chebyshev HDF5 file')
args = parser.parse_args()

project_root = get_project_root()
data_dir = project_root / "data" / "Spectral-Solver"
fig_dir = project_root / "figures" / "Spectral-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

# Determine file paths (use args or defaults)
if args.legendre:
    legendre_path = Path(args.legendre)
else:
    legendre_path = data_dir / f"LDC_Spectral_Re{int(args.Re)}.h5"

if args.chebyshev:
    chebyshev_path = Path(args.chebyshev)
else:
    chebyshev_path = data_dir / f"LDC_Spectral_Chebyshev_Re{int(args.Re)}.h5"

# Validate paths exist
if not legendre_path.exists():
    raise FileNotFoundError(f"Legendre solution not found: {legendre_path}")
if not chebyshev_path.exists():
    raise FileNotFoundError(f"Chebyshev solution not found: {chebyshev_path}")

# Load both solutions
plotter_leg = LDCPlotter(legendre_path)
plotter_cheb = LDCPlotter(chebyshev_path)

validator_leg = GhiaValidator(h5_path=legendre_path, Re=args.Re)
validator_cheb = GhiaValidator(h5_path=chebyshev_path, Re=args.Re)

print(f"Loaded solutions for Re={args.Re}:")
print(f"  Legendre:  {legendre_path.name}")
print(f"  Chebyshev: {chebyshev_path.name}")

# %%
# Ghia Validation Comparison
# ---------------------------
# Side-by-side Ghia benchmark comparison

# Extract centerline data
y_leg, u_leg = validator_leg._extract_centerline_u()
x_leg, v_leg = validator_leg._extract_centerline_v()
y_cheb, u_cheb = validator_cheb._extract_centerline_u()
x_cheb, v_cheb = validator_cheb._extract_centerline_v()

# Compute errors
errors_leg = validator_leg.compute_errors()
errors_cheb = validator_cheb.compute_errors()

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# U-velocity along vertical centerline
ax1.scatter(validator_leg.ghia_u, validator_leg.ghia_y,
           marker='o', s=80, c='black', label='Ghia et al. (1982)',
           zorder=3, edgecolors='white', linewidths=1.5)
ax1.plot(u_leg, y_leg, 'b-', linewidth=2.5,
         label=f'Legendre (RMS={errors_leg["u_rms"]:.4f})', alpha=0.8)
ax1.plot(u_cheb, y_cheb, 'r--', linewidth=2.5,
         label=f'Chebyshev (RMS={errors_cheb["u_rms"]:.4f})', alpha=0.8)
ax1.set_xlabel('u-velocity', fontsize=13, fontweight='bold')
ax1.set_ylabel('y', fontsize=13, fontweight='bold')
ax1.set_title('U-Velocity along Vertical Centerline (x=0.5)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, framealpha=0.95)
ax1.tick_params(labelsize=11)

# V-velocity along horizontal centerline
ax2.scatter(validator_leg.ghia_x, validator_leg.ghia_v,
           marker='o', s=80, c='black', label='Ghia et al. (1982)',
           zorder=3, edgecolors='white', linewidths=1.5)
ax2.plot(x_leg, v_leg, 'b-', linewidth=2.5,
         label=f'Legendre', alpha=0.8)
ax2.plot(x_cheb, v_cheb, 'r--', linewidth=2.5,
         label=f'Chebyshev', alpha=0.8)
ax2.set_xlabel('x', fontsize=13, fontweight='bold')
ax2.set_ylabel('v-velocity', fontsize=13, fontweight='bold')
ax2.set_title('V-Velocity along Horizontal Centerline (y=0.5)',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, framealpha=0.95)
ax2.tick_params(labelsize=11)

grid_size = f"{len(np.unique(plotter_leg.x))}×{len(np.unique(plotter_leg.y))}"
plt.suptitle(f'Ghia Benchmark Validation: Legendre vs Chebyshev\nRe={int(args.Re)}, {grid_size} grid',
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(fig_dir / f"comparison_ghia_validation_Re{int(args.Re)}.pdf", bbox_inches='tight')
print(f"  ✓ Ghia validation comparison saved")

# %%
# Convergence History Comparison
# -------------------------------
# Compare convergence behavior between Legendre and Chebyshev

plotter_leg.plot_convergence(output_path=fig_dir / f"legendre_convergence_Re{int(args.Re)}.pdf")
print(f"  ✓ Legendre convergence saved")

plotter_cheb.plot_convergence(output_path=fig_dir / f"chebyshev_convergence_Re{int(args.Re)}.pdf")
print(f"  ✓ Chebyshev convergence saved")

