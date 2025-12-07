#!/usr/bin/env python
"""Debug the subtraction method - check if singular solution is correct."""

import numpy as np
import sys
sys.path.insert(0, "src")

from solvers.spectral.operators.corner import SubtractionTreatment


def test_singular_solution():
    """Test properties of the Moffatt singular solution."""
    print("=" * 70)
    print("Testing Moffatt Singular Solution Properties")
    print("=" * 70)

    sub = SubtractionTreatment()
    Lx, Ly = 1.0, 1.0

    # Test points near left corner (0, Ly)
    print("\n1. Test at LEFT corner (0, 1):")
    print("-" * 40)

    # Along the lid (y = Ly, varying x)
    x_lid = np.array([0.0, 0.01, 0.05, 0.1, 0.2])
    y_lid = np.full_like(x_lid, Ly)
    u_s_lid, v_s_lid = sub.get_singular_velocity(x_lid, y_lid, Lx, Ly)

    print("\nAlong lid (y=1, varying x):")
    print("  x      u_s        v_s")
    for i in range(len(x_lid)):
        print(f"  {x_lid[i]:.2f}   {u_s_lid[i]:10.4f}   {v_s_lid[i]:10.4f}")

    # Along the left wall (x = 0, varying y)
    x_wall = np.zeros(5)
    y_wall = np.array([1.0, 0.99, 0.95, 0.9, 0.8])
    u_s_wall, v_s_wall = sub.get_singular_velocity(x_wall, y_wall, Lx, Ly)

    print("\nAlong left wall (x=0, varying y):")
    print("  y      u_s        v_s")
    for i in range(len(y_wall)):
        print(f"  {y_wall[i]:.2f}   {u_s_wall[i]:10.4f}   {v_s_wall[i]:10.4f}")

    # Test velocity BC: on lid, total u should be 1, v should be 0
    print("\n2. Checking boundary conditions:")
    print("-" * 40)
    print("\nFor correct subtraction, on the lid:")
    print("  u_c = V_lid - u_s should give smooth u_c")
    print("  v_c = -v_s should give smooth v_c")

    lid_velocity = 1.0
    u_lid_bc, v_lid_bc = sub.get_lid_velocity(x_lid, y_lid, lid_velocity, Lx, Ly)

    print("\nLid BC (u_c = 1 - u_s, v_c = -v_s):")
    print("  x      u_c        v_c")
    for i in range(len(x_lid)):
        print(f"  {x_lid[i]:.2f}   {u_lid_bc[i]:10.4f}   {v_lid_bc[i]:10.4f}")

    # Check: at corner, u_c should be finite, not singular
    print(f"\nAt corner (x=0): u_c = {u_lid_bc[0]:.4f}")
    print("  (Should be finite, not 1.0 or inf)")

    # Test divergence: ∇·u_s should be 0 (streamfunction property)
    print("\n3. Checking divergence of u_s (should be 0):")
    print("-" * 40)

    # Use finite differences to estimate divergence
    eps = 1e-6
    test_x = np.array([0.1, 0.1, 0.1+eps, 0.1-eps, 0.1, 0.1])
    test_y = np.array([0.9, 0.9, 0.9, 0.9, 0.9+eps, 0.9-eps])
    u_test, v_test = sub.get_singular_velocity(test_x, test_y, Lx, Ly)

    du_dx = (u_test[2] - u_test[3]) / (2 * eps)
    dv_dy = (v_test[4] - v_test[5]) / (2 * eps)
    div_us = du_dx + dv_dy

    print(f"At (0.1, 0.9): du_s/dx = {du_dx:.6e}, dv_s/dy = {dv_dy:.6e}")
    print(f"              div(u_s) = {div_us:.6e}")
    print(f"  (Should be ~0 if from streamfunction)")

    # Check analytical derivatives vs finite differences
    print("\n4. Checking analytical derivatives:")
    print("-" * 40)

    x_test = np.array([0.1])
    y_test = np.array([0.9])
    dus_dx, dus_dy, dvs_dx, dvs_dy = sub.get_singular_velocity_derivatives(
        x_test, y_test, Lx, Ly
    )

    # Finite difference estimates
    u_px, _ = sub.get_singular_velocity(np.array([0.1+eps]), y_test, Lx, Ly)
    u_mx, _ = sub.get_singular_velocity(np.array([0.1-eps]), y_test, Lx, Ly)
    u_py, _ = sub.get_singular_velocity(x_test, np.array([0.9+eps]), Lx, Ly)
    u_my, _ = sub.get_singular_velocity(x_test, np.array([0.9-eps]), Lx, Ly)
    _, v_px = sub.get_singular_velocity(np.array([0.1+eps]), y_test, Lx, Ly)
    _, v_mx = sub.get_singular_velocity(np.array([0.1-eps]), y_test, Lx, Ly)
    _, v_py = sub.get_singular_velocity(x_test, np.array([0.9+eps]), Lx, Ly)
    _, v_my = sub.get_singular_velocity(x_test, np.array([0.9-eps]), Lx, Ly)

    dus_dx_fd = (u_px[0] - u_mx[0]) / (2*eps)
    dus_dy_fd = (u_py[0] - u_my[0]) / (2*eps)
    dvs_dx_fd = (v_px[0] - v_mx[0]) / (2*eps)
    dvs_dy_fd = (v_py[0] - v_my[0]) / (2*eps)

    print(f"At (0.1, 0.9):")
    print(f"  du_s/dx: analytical={dus_dx[0]:.6f}, FD={dus_dx_fd:.6f}, diff={abs(dus_dx[0]-dus_dx_fd):.2e}")
    print(f"  du_s/dy: analytical={dus_dy[0]:.6f}, FD={dus_dy_fd:.6f}, diff={abs(dus_dy[0]-dus_dy_fd):.2e}")
    print(f"  dv_s/dx: analytical={dvs_dx[0]:.6f}, FD={dvs_dx_fd:.6f}, diff={abs(dvs_dx[0]-dvs_dx_fd):.2e}")
    print(f"  dv_s/dy: analytical={dvs_dy[0]:.6f}, FD={dvs_dy_fd:.6f}, diff={abs(dvs_dy[0]-dvs_dy_fd):.2e}")

    # Check RIGHT corner (Lx, Ly)
    print("\n5. Test at RIGHT corner (1, 1):")
    print("-" * 40)

    x_right = np.array([1.0, 0.99, 0.95, 0.9, 0.8])
    y_right = np.full_like(x_right, Ly)
    u_s_right, v_s_right = sub.get_singular_velocity(x_right, y_right, Lx, Ly)

    print("\nAlong lid approaching right corner:")
    print("  x      u_s        v_s")
    for i in range(len(x_right)):
        print(f"  {x_right[i]:.2f}   {u_s_right[i]:10.4f}   {v_s_right[i]:10.4f}")


if __name__ == "__main__":
    test_singular_solution()
