"""Isolated tests for FFT transfer operators.

Tests prolongation and restriction operators independently
to verify they work correctly before using in multigrid.
"""

import numpy as np
from solvers.spectral.operators.transfer_operators import (
    FFTProlongation,
    FFTRestriction,
    PolynomialProlongation,
    InjectionRestriction,
)


def chebyshev_lobatto_nodes(N):
    """Chebyshev-Gauss-Lobatto nodes: x_j = cos(πj/N) for j=0,...,N."""
    j = np.arange(N + 1)
    return np.cos(np.pi * j / N)


# Test functions
def poly4(x):
    """Polynomial: x^4 + 2x^2 + 1 (degree 4, exact for N>=4)"""
    return x**4 + 2*x**2 + 1

def poly2(x):
    """Polynomial: x^2 + x + 1 (degree 2, exact for N>=2)"""
    return x**2 + x + 1

def smooth_func(x):
    """Smooth function: sin(πx) (not polynomial, tests interpolation accuracy)"""
    return np.sin(np.pi * x)


def test_prolongation_1d():
    """Test 1D prolongation operators."""
    print("\n" + "=" * 60)
    print("TEST 1: 1D Prolongation (coarse -> fine)")
    print("=" * 60)

    fft_prolong = FFTProlongation()
    poly_prolong = PolynomialProlongation()

    test_cases = [
        ("poly2 (degree 2)", poly2, 4, 8),
        ("poly4 (degree 4)", poly4, 8, 16),
        ("poly4 (degree 4)", poly4, 16, 32),
        ("smooth sin(πx)", smooth_func, 8, 16),
        ("smooth sin(πx)", smooth_func, 16, 32),
    ]

    all_passed = True
    for name, func, N_c, N_f in test_cases:
        x_c = chebyshev_lobatto_nodes(N_c)
        x_f = chebyshev_lobatto_nodes(N_f)

        f_c = func(x_c)
        f_f_exact = func(x_f)

        f_fft = fft_prolong.prolongate_1d(f_c, N_f + 1)
        f_poly = poly_prolong.prolongate_1d(f_c, N_f + 1)

        err_fft = np.max(np.abs(f_fft - f_f_exact))
        err_poly = np.max(np.abs(f_poly - f_f_exact))

        # For polynomials, expect machine precision
        # For smooth functions, expect spectral accuracy (depends on N)
        is_poly = "poly" in name
        # Spectral convergence: error ~ exp(-αN) for smooth functions
        tol = 1e-12 if is_poly else 1e-3

        passed = err_fft < tol and err_poly < tol
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"\n{name} (N={N_c} -> N={N_f}):")
        print(f"  FFT error:  {err_fft:.2e} {'(OK)' if err_fft < tol else '(FAIL)'}")
        print(f"  Poly error: {err_poly:.2e} {'(OK)' if err_poly < tol else '(FAIL)'}")
        print(f"  Boundary values: exact={f_f_exact[0]:.6f}, FFT={f_fft[0]:.6f}, Poly={f_poly[0]:.6f}")
        print(f"  {status}")

    return all_passed


def test_restriction_1d():
    """Test 1D restriction operators."""
    print("\n" + "=" * 60)
    print("TEST 2: 1D Restriction (fine -> coarse)")
    print("=" * 60)

    fft_restrict = FFTRestriction()
    inj_restrict = InjectionRestriction()

    test_cases = [
        ("poly2 (degree 2)", poly2, 8, 4),
        ("poly4 (degree 4)", poly4, 16, 8),
        ("poly4 (degree 4)", poly4, 32, 16),
        ("smooth sin(πx)", smooth_func, 16, 8),
        ("smooth sin(πx)", smooth_func, 32, 16),
    ]

    all_passed = True
    for name, func, N_f, N_c in test_cases:
        x_f = chebyshev_lobatto_nodes(N_f)
        x_c = chebyshev_lobatto_nodes(N_c)

        f_f = func(x_f)
        f_c_exact = func(x_c)

        f_fft = fft_restrict.restrict_1d(f_f, N_c + 1)
        f_inj = inj_restrict.restrict_1d(f_f, N_c + 1)

        err_fft = np.max(np.abs(f_fft - f_c_exact))
        err_inj = np.max(np.abs(f_inj - f_c_exact))

        is_poly = "poly" in name
        tol = 1e-12 if is_poly else 1e-4

        passed = err_fft < tol
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"\n{name} (N={N_f} -> N={N_c}):")
        print(f"  FFT error:       {err_fft:.2e} {'(OK)' if err_fft < tol else '(FAIL)'}")
        print(f"  Injection error: {err_inj:.2e} (reference)")
        print(f"  Boundary values: exact={f_c_exact[0]:.6f}, FFT={f_fft[0]:.6f}, Inj={f_inj[0]:.6f}")
        print(f"  {status}")

    return all_passed


def test_roundtrip():
    """Test round-trip: prolongate then restrict should preserve polynomial."""
    print("\n" + "=" * 60)
    print("TEST 3: Round-trip (coarse -> fine -> coarse)")
    print("=" * 60)

    fft_prolong = FFTProlongation()
    fft_restrict = FFTRestriction()
    inj_restrict = InjectionRestriction()

    all_passed = True
    for N_c in [4, 8, 16]:
        N_f = 2 * N_c
        x_c = chebyshev_lobatto_nodes(N_c)

        f_c = poly4(x_c)

        # Prolongate to fine
        f_f = fft_prolong.prolongate_1d(f_c, N_f + 1)

        # Restrict back to coarse
        f_c_fft = fft_restrict.restrict_1d(f_f, N_c + 1)
        f_c_inj = inj_restrict.restrict_1d(f_f, N_c + 1)

        err_fft = np.max(np.abs(f_c_fft - f_c))
        err_inj = np.max(np.abs(f_c_inj - f_c))

        passed = err_fft < 1e-12 and err_inj < 1e-12
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"\nN={N_c} -> N={N_f} -> N={N_c}:")
        print(f"  FFT round-trip error:       {err_fft:.2e}")
        print(f"  Injection round-trip error: {err_inj:.2e}")
        print(f"  {status}")

    return all_passed


def test_2d_prolongation():
    """Test 2D prolongation."""
    print("\n" + "=" * 60)
    print("TEST 4: 2D Prolongation")
    print("=" * 60)

    fft_prolong = FFTProlongation()
    poly_prolong = PolynomialProlongation()

    def func_2d(X, Y):
        return X**2 + Y**2 + X*Y + 1

    all_passed = True
    for N_c in [4, 8]:
        N_f = 2 * N_c

        x_c = chebyshev_lobatto_nodes(N_c)
        x_f = chebyshev_lobatto_nodes(N_f)

        X_c, Y_c = np.meshgrid(x_c, x_c, indexing='ij')
        X_f, Y_f = np.meshgrid(x_f, x_f, indexing='ij')

        f_c = func_2d(X_c, Y_c)
        f_f_exact = func_2d(X_f, Y_f)

        f_fft = fft_prolong.prolongate_2d(f_c, (N_f + 1, N_f + 1))
        f_poly = poly_prolong.prolongate_2d(f_c, (N_f + 1, N_f + 1))

        err_fft = np.max(np.abs(f_fft - f_f_exact))
        err_poly = np.max(np.abs(f_poly - f_f_exact))

        passed = err_fft < 1e-12 and err_poly < 1e-12
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"\n2D poly (N={N_c} -> N={N_f}):")
        print(f"  FFT error:  {err_fft:.2e}")
        print(f"  Poly error: {err_poly:.2e}")
        print(f"  {status}")

    return all_passed


def test_2d_restriction():
    """Test 2D restriction."""
    print("\n" + "=" * 60)
    print("TEST 5: 2D Restriction")
    print("=" * 60)

    fft_restrict = FFTRestriction()
    inj_restrict = InjectionRestriction()

    def func_2d(X, Y):
        return X**2 + Y**2 + X*Y + 1

    all_passed = True
    for N_f in [8, 16]:
        N_c = N_f // 2

        x_f = chebyshev_lobatto_nodes(N_f)
        x_c = chebyshev_lobatto_nodes(N_c)

        X_f, Y_f = np.meshgrid(x_f, x_f, indexing='ij')
        X_c, Y_c = np.meshgrid(x_c, x_c, indexing='ij')

        f_f = func_2d(X_f, Y_f)
        f_c_exact = func_2d(X_c, Y_c)

        f_fft = fft_restrict.restrict_2d(f_f, (N_c + 1, N_c + 1))
        f_inj = inj_restrict.restrict_2d(f_f, (N_c + 1, N_c + 1))

        err_fft = np.max(np.abs(f_fft - f_c_exact))
        err_inj = np.max(np.abs(f_inj - f_c_exact))

        passed = err_fft < 1e-12 and err_inj < 1e-12
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"\n2D poly (N={N_f} -> N={N_c}):")
        print(f"  FFT error:       {err_fft:.2e}")
        print(f"  Injection error: {err_inj:.2e}")
        print(f"  {status}")

    return all_passed


def test_multigrid_hierarchy():
    """Test operators on typical multigrid grid sequences."""
    print("\n" + "=" * 60)
    print("TEST 6: Multigrid Hierarchy Grids")
    print("=" * 60)

    fft_prolong = FFTProlongation()
    fft_restrict = FFTRestriction()

    # Typical multigrid sequences from paper
    hierarchies = [
        [4, 8, 16],      # 3 levels, fine N=16
        [8, 16, 32],     # 3 levels, fine N=32
        [12, 24, 48],    # 3 levels, fine N=48
        [16, 32, 64],    # 3 levels, fine N=64
    ]

    all_passed = True
    for levels in hierarchies:
        print(f"\nHierarchy: {levels}")

        # Test prolongation through all levels
        for i in range(len(levels) - 1):
            N_c, N_f = levels[i], levels[i+1]
            x_c = chebyshev_lobatto_nodes(N_c)
            x_f = chebyshev_lobatto_nodes(N_f)

            f_c = poly4(x_c)
            f_f_exact = poly4(x_f)

            f_f = fft_prolong.prolongate_1d(f_c, N_f + 1)
            err = np.max(np.abs(f_f - f_f_exact))

            passed = err < 1e-12
            all_passed = all_passed and passed
            status = "✓" if passed else "✗"

            print(f"  Prolong N={N_c} -> N={N_f}: error={err:.2e} {status}")

        # Test restriction through all levels
        for i in range(len(levels) - 1, 0, -1):
            N_f, N_c = levels[i], levels[i-1]
            x_f = chebyshev_lobatto_nodes(N_f)
            x_c = chebyshev_lobatto_nodes(N_c)

            f_f = poly4(x_f)
            f_c_exact = poly4(x_c)

            f_c = fft_restrict.restrict_1d(f_f, N_c + 1)
            err = np.max(np.abs(f_c - f_c_exact))

            passed = err < 1e-12
            all_passed = all_passed and passed
            status = "✓" if passed else "✗"

            print(f"  Restrict N={N_f} -> N={N_c}: error={err:.2e} {status}")

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFER OPERATOR TESTS")
    print("=" * 60)

    results = []
    results.append(("1D Prolongation", test_prolongation_1d()))
    results.append(("1D Restriction", test_restriction_1d()))
    results.append(("Round-trip", test_roundtrip()))
    results.append(("2D Prolongation", test_2d_prolongation()))
    results.append(("2D Restriction", test_2d_restriction()))
    results.append(("Multigrid Hierarchy", test_multigrid_hierarchy()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
