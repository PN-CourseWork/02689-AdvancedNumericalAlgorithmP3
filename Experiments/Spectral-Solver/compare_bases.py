"""
Compare Legendre vs Chebyshev Basis Functions
==============================================

Validate and compare both spectral methods against Ghia benchmark.
"""

from utils import get_project_root
from utils.ghia_validator import GhiaValidator

project_root = get_project_root()
data_dir = project_root / "data" / "Spectral-Solver"

print("="*80)
print(" LEGENDRE vs CHEBYSHEV BASIS COMPARISON")
print("="*80)

# Validate Legendre results
print("\n" + "="*80)
print(" LEGENDRE-GAUSS-LOBATTO BASIS")
print("="*80)
h5_legendre = data_dir / "LDC_Spectral_Re100.h5"
validator_leg = GhiaValidator(h5_path=h5_legendre, Re=100)
validator_leg.print_summary()
errors_leg = validator_leg.compute_errors()

# Validate Chebyshev results
print("\n" + "="*80)
print(" CHEBYSHEV-GAUSS-LOBATTO BASIS (Zhang et al. 2010)")
print("="*80)
h5_chebyshev = data_dir / "LDC_Spectral_Chebyshev_Re100.h5"
validator_cheb = GhiaValidator(h5_path=h5_chebyshev, Re=100)
validator_cheb.print_summary()
errors_cheb = validator_cheb.compute_errors()

# Side-by-side comparison
print("\n" + "="*80)
print(" COMPARISON SUMMARY")
print("="*80)
print(f"{'Metric':<25} {'Legendre':<20} {'Chebyshev':<20} {'Δ (Cheb-Leg)':<15}")
print("-"*80)

metrics = [
    ("U RMS Error", "u_rms"),
    ("U L² Error", "u_l2"),
    ("U L∞ Error", "u_linf"),
    ("V RMS Error", "v_rms"),
    ("V L² Error", "v_l2"),
    ("V L∞ Error", "v_linf"),
]

for name, key in metrics:
    leg_val = errors_leg[key]
    cheb_val = errors_cheb[key]
    diff = cheb_val - leg_val
    print(f"{name:<25} {leg_val:<20.6e} {cheb_val:<20.6e} {diff:<15.6e}")

print("="*80)

# Determine which is better
u_winner = "Legendre" if errors_leg['u_rms'] < errors_cheb['u_rms'] else "Chebyshev"
v_winner = "Legendre" if errors_leg['v_rms'] < errors_cheb['v_rms'] else "Chebyshev"

print(f"\nBest U-velocity accuracy: **{u_winner}**")
print(f"Best V-velocity accuracy: **{v_winner}**")

# Create comparison plots
output_leg = data_dir / "validation_Legendre_Re100.png"
output_cheb = data_dir / "validation_Chebyshev_Re100.png"

validator_leg.plot_validation(output_path=output_leg)
validator_cheb.plot_validation(output_path=output_cheb)

print(f"\nValidation plots saved:")
print(f"  - {output_leg}")
print(f"  - {output_cheb}")
