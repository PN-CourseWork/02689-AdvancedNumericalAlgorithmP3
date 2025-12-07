"""Debug the inner grid differentiation matrix."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from solvers.spectral.basis.spectral import (
    ChebyshevLobattoBasis,
    chebyshev_gauss_lobatto_nodes,
    chebyshev_diff_matrix,
)

print("=" * 60)
print("DEBUG: Inner grid differentiation")
print("=" * 60)

N = 15
basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))

# Full grid nodes
nodes_full = basis.nodes(N + 1)
print(f"Full grid: {N+1} nodes")
print(f"  Nodes: {nodes_full}")

# Inner grid nodes (excluding boundaries)
nodes_inner = nodes_full[1:-1]
print(f"\nInner grid: {N-1} nodes")
print(f"  Nodes: {nodes_inner}")

# Build differentiation matrices
D_full = basis.diff_matrix(nodes_full)
D_inner = basis.diff_matrix(nodes_inner)

print(f"\nFull diff matrix shape: {D_full.shape}")
print(f"Inner diff matrix shape: {D_inner.shape}")

# Test the inner grid differentiation on a polynomial
# f(x) = x^2, df/dx = 2x
f_inner = nodes_inner**2
df_exact_inner = 2 * nodes_inner
df_spectral_inner = D_inner @ f_inner

print(f"\nTest: d/dx[x²] on inner grid")
print(f"  Exact: {df_exact_inner}")
print(f"  Spectral: {df_spectral_inner}")
print(f"  Error: {np.max(np.abs(df_spectral_inner - df_exact_inner)):.2e}")

# Test with a higher-degree polynomial
f_inner = nodes_inner**4
df_exact_inner = 4 * nodes_inner**3
df_spectral_inner = D_inner @ f_inner

print(f"\nTest: d/dx[x⁴] on inner grid")
print(f"  Error: {np.max(np.abs(df_spectral_inner - df_exact_inner)):.2e}")

# Compare with extracting the inner portion from full-grid differentiation
# If we have f on the full grid, we can compute df on full grid, then extract inner
f_full = nodes_full**2
df_full = D_full @ f_full
df_full_inner = df_full[1:-1]  # Extract inner portion

print(f"\n" + "=" * 60)
print("Comparison: Inner-grid diff vs Full-grid diff restricted to inner")
print("=" * 60)

# Method 1: Use inner diff matrix directly on inner values
method1 = D_inner @ f_inner

# Method 2: Use full diff matrix on full values, then extract inner
# But wait - to use the full diff matrix, we need f on the FULL grid
# What should the boundary values be?

# For pressure, we DON'T know the boundary values!
# So using D_inner is the correct approach IF the inner diff matrix is correct.

# The issue is: the inner diff matrix is built using nodes that are NOT CGL nodes
# The inner nodes don't include the endpoints [-1, 1], so they're not CGL nodes!

# Let me check what nodes the inner diff matrix is using:
print(f"\nInner nodes mapped to [-1, 1]:")
a, b = 0.0, 1.0
xi_inner = 2.0 * (nodes_inner - a) / (b - a) - 1.0
print(f"  xi_inner = {xi_inner}")

# These are NOT CGL nodes! CGL nodes have endpoints at ±1
# The Chebyshev differentiation matrix is designed for CGL nodes!

# Let's verify by checking if the inner diff matrix satisfies the row-sum property
# For a proper diff matrix, sum of each row should be zero (derivative of constant = 0)
row_sums = np.sum(D_inner, axis=1)
print(f"\nRow sums of D_inner (should all be ~0):")
print(f"  {row_sums}")
print(f"  Max row sum: {np.max(np.abs(row_sums)):.2e}")

# Check the full diff matrix row sums
row_sums_full = np.sum(D_full, axis=1)
print(f"\nRow sums of D_full (should all be ~0):")
print(f"  Max row sum: {np.max(np.abs(row_sums_full)):.2e}")

print("\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("""
The inner grid differentiation matrix is built using basis.diff_matrix(nodes_inner).
But nodes_inner are NOT Chebyshev-Gauss-Lobatto nodes - they're a subset missing
the endpoints. The Chebyshev diff matrix formula assumes CGL nodes!

This means the inner diff matrix is INCORRECT for spectral differentiation.

CORRECT APPROACH for PN-PN-2:
1. Compute pressure gradient on FULL grid (even though p is only on inner grid)
2. Use extrapolation/padding to get p on boundaries first
3. Or use a different formulation
""")

# Alternative: What if we use Legendre instead of Chebyshev?
# Let's check the Legendre diff matrix
from solvers.spectral.basis.spectral import LegendreLobattoBasis

basis_leg = LegendreLobattoBasis(domain=(0.0, 1.0))
nodes_leg_full = basis_leg.nodes(N + 1)
nodes_leg_inner = nodes_leg_full[1:-1]

D_leg_inner = basis_leg.diff_matrix(nodes_leg_inner)

f_leg_inner = nodes_leg_inner**2
df_leg_exact = 2 * nodes_leg_inner
df_leg_spectral = D_leg_inner @ f_leg_inner

print(f"\nLegendre inner grid test: d/dx[x²]")
print(f"  Error: {np.max(np.abs(df_leg_spectral - df_leg_exact)):.2e}")

row_sums_leg = np.sum(D_leg_inner, axis=1)
print(f"  Max row sum of D_inner: {np.max(np.abs(row_sums_leg)):.2e}")
