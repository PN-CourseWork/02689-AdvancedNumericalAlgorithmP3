"""Quick diagnostic script for the spectral solver."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from solvers.spectral.basis.spectral import (
    ChebyshevLobattoBasis,
    chebyshev_gauss_lobatto_nodes,
    chebyshev_diff_matrix,
)

# =============================================================================
# Test 1: Verify CGL nodes are correct
# =============================================================================
print("=" * 60)
print("TEST 1: Chebyshev-Gauss-Lobatto nodes")
print("=" * 60)

N = 8
nodes_ref = chebyshev_gauss_lobatto_nodes(N + 1)
print(f"Reference CGL nodes for N={N}: x_j = -cos(πj/N)")
print(f"  Nodes: {nodes_ref}")
print(f"  Expected: [-1, ..., 1] (monotonically increasing)")
print(f"  Min: {nodes_ref[0]}, Max: {nodes_ref[-1]}")
print(f"  Monotonic: {np.all(np.diff(nodes_ref) > 0)}")

# Test with basis class
basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
nodes_physical = basis.nodes(N + 1)
print(f"\nPhysical domain [0, 1] nodes:")
print(f"  Nodes: {nodes_physical}")
print(f"  Min: {nodes_physical[0]}, Max: {nodes_physical[-1]}")
print(f"  Monotonic: {np.all(np.diff(nodes_physical) > 0)}")

# =============================================================================
# Test 2: Verify differentiation matrix
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: Differentiation matrix accuracy")
print("=" * 60)

# Test: d/dx[x^2] = 2x
def test_diff_matrix(N, domain=(0.0, 1.0)):
    basis = ChebyshevLobattoBasis(domain=domain)
    nodes = basis.nodes(N + 1)
    D = basis.diff_matrix(nodes)

    # Test function: f(x) = x^2
    f = nodes**2
    df_exact = 2 * nodes
    df_spectral = D @ f

    error = np.max(np.abs(df_spectral - df_exact))
    print(f"  N={N}: max error for d/dx[x²] = {error:.2e}")

    # Test function: f(x) = sin(πx)
    f = np.sin(np.pi * nodes)
    df_exact = np.pi * np.cos(np.pi * nodes)
    df_spectral = D @ f
    error_sin = np.max(np.abs(df_spectral - df_exact))
    print(f"  N={N}: max error for d/dx[sin(πx)] = {error_sin:.2e}")

    return error, error_sin

for N in [8, 16, 32]:
    test_diff_matrix(N)

# =============================================================================
# Test 3: Check the 2D Kronecker product differentiation
# =============================================================================
print("\n" + "=" * 60)
print("TEST 3: 2D Kronecker product differentiation")
print("=" * 60)

N = 8
basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
nodes = basis.nodes(N + 1)
D_1d = basis.diff_matrix(nodes)

# Build 2D diff matrices
Ix = np.eye(N + 1)
Dx = np.kron(D_1d, Ix)  # d/dx
Dy = np.kron(Ix, D_1d)  # d/dy

# Create 2D grid
X, Y = np.meshgrid(nodes, nodes, indexing='ij')

# Test function: f(x,y) = x^2 + y^2
f_2d = X**2 + Y**2
f_flat = f_2d.ravel()

df_dx_exact = 2 * X
df_dy_exact = 2 * Y

df_dx_spectral = (Dx @ f_flat).reshape(X.shape)
df_dy_spectral = (Dy @ f_flat).reshape(Y.shape)

error_dx = np.max(np.abs(df_dx_spectral - df_dx_exact))
error_dy = np.max(np.abs(df_dy_spectral - df_dy_exact))

print(f"Test f(x,y) = x² + y²:")
print(f"  d/dx error: {error_dx:.2e}")
print(f"  d/dy error: {error_dy:.2e}")

# Test Laplacian
Dxx = np.kron(D_1d @ D_1d, Ix)
Dyy = np.kron(Ix, D_1d @ D_1d)
Laplacian = Dxx + Dyy

lap_exact = 4.0  # ∇²(x² + y²) = 2 + 2 = 4
lap_spectral = (Laplacian @ f_flat).reshape(X.shape)
error_lap = np.max(np.abs(lap_spectral - lap_exact))
print(f"  Laplacian error: {error_lap:.2e}")

# =============================================================================
# Test 4: Check boundary conditions
# =============================================================================
print("\n" + "=" * 60)
print("TEST 4: Boundary condition verification")
print("=" * 60)

from solvers.spectral.operators.corner import SmoothingTreatment

N = 15
basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
nodes = basis.nodes(N + 1)
X, Y = np.meshgrid(nodes, nodes, indexing='ij')

corner = SmoothingTreatment(smoothing_width=0.15)

# Test lid boundary (top, y=1)
x_lid = X[:, -1]
y_lid = Y[:, -1]
u_lid, v_lid = corner.get_lid_velocity(x_lid, y_lid, lid_velocity=1.0, Lx=1.0, Ly=1.0)

print(f"Lid velocity (y=1):")
print(f"  u_lid: min={u_lid.min():.4f}, max={u_lid.max():.4f}")
print(f"  v_lid: min={v_lid.min():.4f}, max={v_lid.max():.4f}")
print(f"  u_lid at corners: [{u_lid[0]:.4f}, {u_lid[-1]:.4f}]")
print(f"  u_lid at center: {u_lid[N//2]:.4f}")

# Test wall boundary
x_wall = X[0, :]
y_wall = Y[0, :]
u_wall, v_wall = corner.get_wall_velocity(x_wall, y_wall, Lx=1.0, Ly=1.0)
print(f"\nWall velocity (x=0):")
print(f"  u_wall: {np.all(u_wall == 0)}")
print(f"  v_wall: {np.all(v_wall == 0)}")

# =============================================================================
# Test 5: Run a few iterations and check residual behavior
# =============================================================================
print("\n" + "=" * 60)
print("TEST 5: Quick solver run")
print("=" * 60)

from solvers.spectral.sg import SGSolver

solver = SGSolver(
    Re=100,
    nx=15,
    ny=15,
    tolerance=1e-6,
    max_iterations=100,
    lid_velocity=1.0,
    Lx=1.0,
    Ly=1.0,
    CFL=0.5,
    beta_squared=5.0,
    basis_type="chebyshev",
    corner_treatment="smoothing",
)

# Run more steps manually
n_iters = 50000
print(f"Running {n_iters} iterations...")
for i in range(n_iters):
    u, v, p = solver.step()
    if i % 500 == 0:
        u_res = np.linalg.norm(solver.arrays.u - solver.arrays.u_prev)
        v_res = np.linalg.norm(solver.arrays.v - solver.arrays.v_prev)
        div = solver.arrays.du_dx + solver.arrays.dv_dy
        max_div = np.max(np.abs(div))
        p_2d = solver.arrays.p.reshape(solver.shape_inner)
        print(f"  Iter {i}: u_change={u_res:.4e}, v_change={v_res:.4e}, max_div={max_div:.4e}, p_range=[{p_2d.min():.4f}, {p_2d.max():.4f}]")

# Check solution characteristics
u_2d = solver.arrays.u.reshape(solver.shape_full)
v_2d = solver.arrays.v.reshape(solver.shape_full)

print(f"\nSolution after 100 iterations:")
print(f"  u: min={u_2d.min():.4f}, max={u_2d.max():.4f}")
print(f"  v: min={v_2d.min():.4f}, max={v_2d.max():.4f}")
print(f"  Lid BC u[:,-1]: min={u_2d[:,-1].min():.4f}, max={u_2d[:,-1].max():.4f}")
print(f"  Wall BC u[0,:]: all zero? {np.allclose(u_2d[0,:], 0, atol=1e-10)}")
print(f"  Wall BC u[-1,:]: all zero? {np.allclose(u_2d[-1,:], 0, atol=1e-10)}")
print(f"  Wall BC u[:,0]: all zero? {np.allclose(u_2d[:,0], 0, atol=1e-10)}")

# Check continuity
print(f"\nDivergence check:")
div = solver.arrays.du_dx + solver.arrays.dv_dy
div_2d = div.reshape(solver.shape_full)
print(f"  max |div(u)|: {np.max(np.abs(div_2d)):.4e}")

# Check pressure
p_2d = solver.arrays.p.reshape(solver.shape_inner)
print(f"\nPressure check:")
print(f"  p range: [{p_2d.min():.4f}, {p_2d.max():.4f}]")
print(f"  p mean: {p_2d.mean():.4f}")

# =============================================================================
# Test 6: Compare centerline velocities to Ghia
# =============================================================================
print("\n" + "=" * 60)
print(f"TEST 6: Comparison to Ghia (after {n_iters} iterations)")
print("=" * 60)

import pandas as pd

ghia_u = pd.read_csv("data/validation/ghia/ghia_Re100_u_centerline.csv")
ghia_v = pd.read_csv("data/validation/ghia/ghia_Re100_v_centerline.csv")

# Get centerline u(x=0.5, y)
x_center_idx = np.argmin(np.abs(nodes - 0.5))
u_centerline = u_2d[x_center_idx, :]
y_centerline = Y[x_center_idx, :]

print(f"Vertical centerline at x={nodes[x_center_idx]:.4f}:")
print(f"  y values: {y_centerline}")
print(f"  u values: {u_centerline}")

# Interpolate Ghia to our grid points
u_ghia_interp = np.interp(y_centerline, ghia_u['y'].values, ghia_u['u'].values)
error = np.max(np.abs(u_centerline - u_ghia_interp))
print(f"  Max error vs Ghia: {error:.4f}")

# Get centerline v(x, y=0.5)
y_center_idx = np.argmin(np.abs(nodes - 0.5))
v_centerline = v_2d[:, y_center_idx]
x_centerline = X[:, y_center_idx]

print(f"\nHorizontal centerline at y={nodes[y_center_idx]:.4f}:")
print(f"  x values: {x_centerline}")
print(f"  v values: {v_centerline}")

v_ghia_interp = np.interp(x_centerline, ghia_v['x'].values, ghia_v['v'].values)
error_v = np.max(np.abs(v_centerline - v_ghia_interp))
print(f"  Max error vs Ghia: {error_v:.4f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
