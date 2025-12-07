"""Debug the pressure gradient computation."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis

print("=" * 60)
print("DEBUG: Pressure gradient computation")
print("=" * 60)

N = 15
basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
nodes_full = basis.nodes(N + 1)
nodes_inner = nodes_full[1:-1]

D_full = basis.diff_matrix(nodes_full)

print(f"Full grid: {N+1} nodes")
print(f"Inner grid: {N-1} nodes")

# Test: If pressure is p(x) = x^2 on inner grid, what is dp/dx?
# True dp/dx = 2x

p_inner = nodes_inner**2
dp_dx_exact_full = 2 * nodes_full

# Method 1: Linear extrapolation then differentiate
def extrapolate_to_full(inner_1d, nodes_full):
    """Linear extrapolation to boundaries."""
    n_full = len(nodes_full)
    full = np.zeros(n_full)
    full[1:-1] = inner_1d
    # Linear extrapolation
    full[0] = 2 * full[1] - full[2]
    full[-1] = 2 * full[-2] - full[-3]
    return full

p_full_linear = extrapolate_to_full(p_inner, nodes_full)
dp_dx_linear = D_full @ p_full_linear

print(f"\nMethod 1: Linear extrapolation")
print(f"  p_full boundary: [{p_full_linear[0]:.4f}, {p_full_linear[-1]:.4f}]")
print(f"  Exact boundary: [{nodes_full[0]**2:.4f}, {nodes_full[-1]**2:.4f}]")
print(f"  dp/dx error: {np.max(np.abs(dp_dx_linear - dp_dx_exact_full)):.4e}")

# Method 2: Polynomial fit and evaluate
from numpy.polynomial.chebyshev import chebfit, chebval, chebder

def extrapolate_spectral(inner_1d, nodes_inner, nodes_full):
    """Fit polynomial to inner values, evaluate on full grid."""
    # Map to [-1, 1]
    xi_inner = 2 * nodes_inner - 1
    xi_full = 2 * nodes_full - 1

    # Fit Chebyshev polynomial of degree N-2 (matching inner grid size)
    coeffs = chebfit(xi_inner, inner_1d, deg=len(inner_1d)-1)

    # Evaluate on full grid
    return chebval(xi_full, coeffs)

p_full_spectral = extrapolate_spectral(p_inner, nodes_inner, nodes_full)
dp_dx_spectral = D_full @ p_full_spectral

print(f"\nMethod 2: Spectral extrapolation (Chebyshev polynomial fit)")
print(f"  p_full boundary: [{p_full_spectral[0]:.4f}, {p_full_spectral[-1]:.4f}]")
print(f"  Exact boundary: [{nodes_full[0]**2:.4f}, {nodes_full[-1]**2:.4f}]")
print(f"  dp/dx error: {np.max(np.abs(dp_dx_spectral - dp_dx_exact_full)):.4e}")

# Method 3: Build a proper interpolation matrix from inner to full
# and a derivative matrix that goes from inner to full
def build_interpolation_and_derivative_matrices(nodes_inner, nodes_full):
    """Build matrices that interpolate p from inner to full and compute dp/dx on full."""
    from numpy.polynomial.chebyshev import chebvander

    # Map to [-1, 1]
    xi_inner = 2 * nodes_inner - 1
    xi_full = 2 * nodes_full - 1

    n_inner = len(nodes_inner)
    n_full = len(nodes_full)

    # Vandermonde matrix on inner points
    V_inner = chebvander(xi_inner, n_inner - 1)

    # Vandermonde matrix on full points
    V_full = chebvander(xi_full, n_inner - 1)

    # Interpolation matrix: P_full = I @ P_inner
    # P_inner = V_inner^{-1} @ coeffs, P_full = V_full @ coeffs
    # So I = V_full @ V_inner^{-1}
    Interp = V_full @ np.linalg.solve(V_inner, np.eye(n_inner))

    # For derivative, we need the derivative Vandermonde
    # T_k'(x) = k * U_{k-1}(x) where U is Chebyshev of second kind
    # But easier: just differentiate after interpolation
    # dp/dx = D_full @ (Interp @ p_inner) = (D_full @ Interp) @ p_inner

    return Interp

Interp = build_interpolation_and_derivative_matrices(nodes_inner, nodes_full)
p_full_interp = Interp @ p_inner
dp_dx_interp = D_full @ p_full_interp

print(f"\nMethod 3: Interpolation matrix")
print(f"  p_full boundary: [{p_full_interp[0]:.4f}, {p_full_interp[-1]:.4f}]")
print(f"  Exact boundary: [{nodes_full[0]**2:.4f}, {nodes_full[-1]**2:.4f}]")
print(f"  dp/dx error: {np.max(np.abs(dp_dx_interp - dp_dx_exact_full)):.4e}")

# Test on a more complex function: p = sin(πx)
print("\n" + "=" * 60)
print("Test with p = sin(πx)")
print("=" * 60)

p_inner_sin = np.sin(np.pi * nodes_inner)
dp_dx_exact_sin = np.pi * np.cos(np.pi * nodes_full)

p_full_linear_sin = extrapolate_to_full(p_inner_sin, nodes_full)
dp_dx_linear_sin = D_full @ p_full_linear_sin

p_full_spectral_sin = extrapolate_spectral(p_inner_sin, nodes_inner, nodes_full)
dp_dx_spectral_sin = D_full @ p_full_spectral_sin

p_full_interp_sin = Interp @ p_inner_sin
dp_dx_interp_sin = D_full @ p_full_interp_sin

print(f"Method 1 (Linear): dp/dx error = {np.max(np.abs(dp_dx_linear_sin - dp_dx_exact_sin)):.4e}")
print(f"Method 2 (Spectral): dp/dx error = {np.max(np.abs(dp_dx_spectral_sin - dp_dx_exact_sin)):.4e}")
print(f"Method 3 (Interp): dp/dx error = {np.max(np.abs(dp_dx_interp_sin - dp_dx_exact_sin)):.4e}")

# Check the boundary extrapolation errors
print(f"\nBoundary p values for sin(πx):")
print(f"  Linear:   [{p_full_linear_sin[0]:.6f}, {p_full_linear_sin[-1]:.6f}]")
print(f"  Spectral: [{p_full_spectral_sin[0]:.6f}, {p_full_spectral_sin[-1]:.6f}]")
print(f"  Interp:   [{p_full_interp_sin[0]:.6f}, {p_full_interp_sin[-1]:.6f}]")
print(f"  Exact:    [{np.sin(np.pi*nodes_full[0]):.6f}, {np.sin(np.pi*nodes_full[-1]):.6f}]")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
The interpolation matrix approach gives spectral accuracy.
We should use this instead of linear extrapolation.

But wait - there's another issue: the linear extrapolation formula
uses 2*f[1] - f[2] which is correct for LINEAR functions but not
for higher-order polynomials.

For spectral accuracy, we need to either:
1. Use the Chebyshev polynomial fit (Method 2)
2. Build proper interpolation matrices (Method 3)
""")
