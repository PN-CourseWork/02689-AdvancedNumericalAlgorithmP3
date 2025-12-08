# FAS (Full Approximation Storage) V-Cycle Multigrid Implementation Plan

## Overview

This document outlines the implementation of a new `fas.py` solver that closely follows the Zhang & Xi (2010) paper: *"An explicit Chebyshev pseudospectral multigrid method for incompressible Navier-Stokes equations"*.

The goal is to create a clean, paper-faithful implementation that achieves the reported **~90% time reduction** (10x speedup) with 3-level multigrid.

---

## Key Design Decisions

### 1. Algorithm Parameters (Paper-Faithful)

| Parameter | Paper Value | Current VMG | New FAS |
|-----------|-------------|-------------|---------|
| Pre-smoothing steps | 1 per level | 10 | **1** |
| Post-smoothing steps | 0 | 10 | **0** |
| Damping factor | 1.0 (implicit) | 0.8 | **1.0** (no damping) |
| Convergence criterion | Continuity RMS | Solution change | **Continuity RMS** |
| Notation | `N-VMG-111` | - | `N-FAS-L` (L=levels) |

### 2. Convergence Criterion

**Paper (Section 4):**
```
E_RMS = sqrt( Σᵢⱼ (∂u/∂x + ∂v/∂y)²ᵢⱼ / ((Nx-1)(Ny-1)) ) < ε
```
- Computed on **inner grid only** (Nx-1 × Ny-1 points)
- This is the **continuity equation residual** (divergence)
- Tolerance: ε = 10⁻⁴ for cavity flow, 10⁻¹⁰ for test problems

### 3. RK4 Time Stepping (Same as Paper)

```python
# 4-stage explicit RK4 (Eq. 7)
coefficients = [1/4, 1/3, 1/2, 1]

φ⁽¹⁾ = φⁿ + (1/4) Δτ R(φⁿ)
φ⁽²⁾ = φⁿ + (1/3) Δτ R(φ⁽¹⁾)
φ⁽³⁾ = φⁿ + (1/2) Δτ R(φ⁽²⁾)
φⁿ⁺¹ = φⁿ + Δτ R(φ⁽³⁾)
```

### 4. Transfer Operators

| Operation | Method | Description |
|-----------|--------|-------------|
| **Restrict solution** | Direct injection | Coarse GLL nodes are subset of fine |
| **Restrict residual** | FFT + truncation | High-frequency filtering (Eq. in Section 3.3) |
| **Prolongate correction** | FFT (DCT) | Chebyshev interpolation (Eq. 10-11) |

### 5. Grid Hierarchy

- Full coarsening: Nᶜ = Nᶠ / 2
- Coarsest grid: N ≥ 12 (need to resolve physics)
- Example for N=64: levels [16, 32, 64] (3 levels)

---

## V-Cycle FAS Algorithm

```
V-CYCLE-FAS(level_idx):
    level = levels[level_idx]

    # 1. Pre-smoothing (1 RK4 step per paper)
    for i = 1 to pre_smooth:
        RK4_step(level)

    # 2. If not coarsest level:
    if level_idx > 0:
        coarse = levels[level_idx - 1]

        # 2a. Compute fine grid residual
        R_h = compute_residuals(u_h, v_h, p_h)

        # 2b. Restrict residual (FFT with truncation)
        # Zero boundaries BEFORE restriction
        R_h[boundaries] = 0
        I_r = restrict_residual(R_h)  # FFT-based

        # 2c. Restrict solution (direct injection)
        u_H = restrict_solution(u_h)  # injection

        # 2d. Compute tau correction (FAS key step)
        R_H' = compute_residuals(u_H, v_H, p_H)  # Coarse residual at restricted solution
        tau = I_r - R_H'  # Correction to preserve fine-grid information
        tau[boundaries] = 0

        # 2e. Solve on coarse grid with modified RHS
        set_tau_correction(tau)
        V-CYCLE-FAS(level_idx - 1)  # Recursion
        clear_tau_correction()

        # 2f. Compute correction
        e_H = v_H - u_H  # New solution minus old
        e_H[boundaries] = 0

        # 2g. Prolongate correction (FFT-based)
        e_h = prolongate(e_H)
        e_h[boundaries] = 0

        # 2h. Apply correction (no damping per paper)
        u_h = u_h + e_h  # damping = 1.0

        # 2i. Re-enforce boundary conditions
        enforce_BCs(u_h, v_h)

    # 3. Post-smoothing (0 per paper - skip entirely)
    # Paper: "postsmoothing is unnecessary since in most cases
    #         it brings no improvements but sometimes brings instabilities"
```

---

## File Structure

```
src/solvers/spectral/
├── fas.py              # NEW: Clean FAS implementation
├── sg.py               # Single grid (base class)
├── fsg.py              # Existing FSG
├── vmg.py              # Existing VMG (keep for comparison)
└── multigrid/
    └── fsg.py          # Shared utilities (SpectralLevel, MultigridSmoother)

conf/solver/spectral/
├── fas.yaml            # NEW: FAS configuration
└── ...
```

---

## Class Design

### FASLevel (Lightweight Data Structure)

```python
@dataclass
class FASLevel:
    """Data for one multigrid level."""
    n: int                    # Polynomial order
    shape_full: Tuple[int, int]
    shape_inner: Tuple[int, int]

    # Grids
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    dx_min: float
    dy_min: float

    # Differentiation matrices
    Dx: np.ndarray
    Dy: np.ndarray
    Laplacian: np.ndarray
    Interp_x: np.ndarray      # Inner->full interpolation
    Interp_y: np.ndarray

    # Solution arrays (flattened)
    u: np.ndarray             # Full grid
    v: np.ndarray             # Full grid
    p: np.ndarray             # Inner grid

    # RK4 staging
    u_stage: np.ndarray
    v_stage: np.ndarray
    p_stage: np.ndarray

    # Residuals
    R_u: np.ndarray           # Full grid
    R_v: np.ndarray           # Full grid
    R_p: np.ndarray           # Inner grid (continuity)

    # FAS tau correction (set during coarse solve)
    tau_u: np.ndarray = None
    tau_v: np.ndarray = None
    tau_p: np.ndarray = None
```

### FASSolver Class

```python
class FASSolver(SGSolver):
    """V-cycle FAS Multigrid solver following Zhang & Xi (2010)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Build multigrid hierarchy
        self.levels = self._build_hierarchy()
        self.n_levels = len(self.levels)

        # FAS parameters (paper defaults)
        self.pre_smooth = self.params.pre_smooth    # Default: 1
        self.post_smooth = self.params.post_smooth  # Default: 0

        # Transfer operators
        self.transfer_ops = create_transfer_operators("fft", "fft")

    def solve(self, tolerance=None, max_iter=None):
        """Main solve loop using V-cycles."""
        # Initialize finest level
        finest = self.levels[-1]
        finest.u[:] = 0
        finest.v[:] = 0
        finest.p[:] = 0
        self._enforce_lid_bc(finest)

        for cycle in range(max_cycles):
            # Perform one V-cycle
            self._vcycle(level_idx=self.n_levels - 1)

            # Check convergence using continuity RMS
            erms = self._compute_continuity_rms(finest)

            if erms < tolerance:
                return finest, cycle + 1, True

        return finest, max_cycles, False

    def _vcycle(self, level_idx: int):
        """Perform one V-cycle starting at level_idx."""
        # Implementation as described above
        ...

    def _compute_continuity_rms(self, level) -> float:
        """Compute RMS of divergence on inner grid (paper's E_RMS)."""
        # Compute divergence
        du_dx = level.Dx @ level.u
        dv_dy = level.Dy @ level.v
        div_full = du_dx + dv_dy

        # Extract inner grid
        div_2d = div_full.reshape(level.shape_full)
        div_inner = div_2d[1:-1, 1:-1].ravel()

        # RMS
        n_inner = len(div_inner)
        return np.sqrt(np.sum(div_inner**2) / n_inner)
```

---

## Configuration (fas.yaml)

```yaml
# @package solver
# FAS V-cycle Multigrid (Zhang & Xi 2010)
defaults:
  - /solver/spectral/sg

_target_: solvers.spectral.fas.FASSolver
name: spectral_fas

# Multigrid settings
multigrid: fas
n_levels: 3               # Max levels (actual determined by coarsest_n)
coarsest_n: 12            # Minimum N for coarsest grid

# FAS parameters (paper-faithful defaults)
pre_smooth: 1             # Paper uses 1
post_smooth: 0            # Paper says "unnecessary" and "sometimes brings instabilities"

# Transfer operators
prolongation_method: fft
restriction_method: fft

# Convergence criterion: continuity RMS (paper's E_RMS)
convergence_criterion: continuity_rms
```

---

## Key Differences from Current VMG

| Aspect | Current VMG | New FAS |
|--------|-------------|---------|
| Pre-smoothing | 10 | **1** |
| Post-smoothing | 10 | **0** |
| Damping | 0.8 | **1.0** (none) |
| Convergence | Solution change | **Continuity RMS** |
| Code structure | Uses MultigridSmoother class | Simpler inline methods |
| Tau zeroing | At boundaries | Same but cleaner |

---

## Testing Plan

1. **Accuracy test**: Compare with SG solver on N=48, Re=100
   - Should match to 6+ digits

2. **Speedup test**: Compare wall time vs SG
   - 2-level: expect ~50% reduction
   - 3-level: expect ~90% reduction (10x speedup)

3. **Convergence test**: Plot E_RMS vs V-cycles
   - Should show rapid convergence

4. **Ghia benchmark**: Validate against Ghia et al. (1982) centerline velocities

---

## Questions for Discussion

1. **Coarsest grid solve**: Should we solve to full convergence on coarsest, or just do `pre_smooth` iterations like other levels?
   - Paper notation "VMG-111" suggests same iterations on all levels
   - But some implementations solve coarsest exactly

2. **Corner treatment on coarse grids**: Current VMG uses smoothing on N<8. Should we:
   - Keep this behavior?
   - Always use smoothing for simplicity?
   - Match fine level treatment?

3. **Reuse existing code**: Should we reuse `SpectralLevel` from `multigrid/fsg.py` or create fresh `FASLevel`?
   - Reuse: Less code duplication
   - Fresh: Cleaner, simpler, paper-focused

4. **CFL number**: Paper uses CFL=2.5 but notes smaller values (0.3-0.5) for small N.
   - Should we auto-adapt CFL based on N?
   - Or use fixed CFL=2.0 from current config?

---

## Implementation Order

1. [ ] Create `FASLevel` dataclass (or reuse `SpectralLevel`)
2. [ ] Create `FASSolver` class skeleton inheriting from `SGSolver`
3. [ ] Implement `_build_hierarchy()`
4. [ ] Implement `_rk4_step()` for single level
5. [ ] Implement `_compute_continuity_rms()` convergence criterion
6. [ ] Implement `_vcycle()` with FAS algorithm
7. [ ] Create `fas.yaml` config
8. [ ] Test on simple cases
9. [ ] Benchmark speedup vs SG
10. [ ] Validate against Ghia benchmark

---

## References

- Zhang, W., Zhang, C.H., Xi, G. (2010). "An explicit Chebyshev pseudospectral multigrid method for incompressible Navier–Stokes equations." *Computers & Fluids*, 39(1), 178-188.
- Botella, O., Peyret, R. (1998). "Benchmark spectral results on the lid-driven cavity flow." *Computers & Fluids*, 27(4), 421-433.
- Ghia, U., Ghia, K.N., Shin, C.T. (1982). "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." *Journal of Computational Physics*, 48(3), 387-411.
