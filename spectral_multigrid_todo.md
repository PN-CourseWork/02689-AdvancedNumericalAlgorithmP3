# Spectral Multigrid Implementation Plan

Based on Zhang & Xi (2010): "An explicit Chebyshev pseudospectral multigrid method for incompressible Navier-Stokes equations"

## Overview

Three multigrid strategies to implement incrementally:

| Method | Speedup | Complexity |
|--------|---------|------------|
| **FSG** (Full Single Grid) | ~10-18x | Simplest |
| **VMG** (V-cycle Multigrid) | ~50-90% reduction | Medium |
| **FMG** (Full Multigrid) | ~100x | FSG + VMG combined |

---

## Phase 1: Foundation & FSG

### 1.1 SpectralLevel Class
- [ ] Create `SpectralLevel` dataclass/class holding:
  - `n`: polynomial order for this level
  - `basis`: basis object (Chebyshev/Legendre)
  - `x, y`: 1D node arrays
  - `X, Y`: 2D meshgrid arrays (full grid)
  - `X_inner, Y_inner`: 2D meshgrid arrays (inner grid for pressure)
  - `Dx, Dy`: differentiation matrices (full grid)
  - `Dx_inner, Dy_inner`: differentiation matrices (inner grid)
  - `dx_min, dy_min`: minimum grid spacing (for CFL)
  - Field arrays: `u, v, p` and RK4 stage storage

### 1.2 Grid Hierarchy Construction
- [ ] Define `build_hierarchy(n_fine, n_levels)`:
  - Coarsening: N^c = N^f / 2 (full coarsening)
  - Example: 64 → 32 → 16 (3 levels)
- [ ] Verify coarse Lobatto nodes are subsets of fine nodes
- [ ] Construct `SpectralLevel` for each level

### 1.3 Prolongation Operator (Solution Interpolation)
- [ ] Implement `prolongate_solution(u_coarse, level_coarse, level_fine)`:
  - For velocities (u, v): FFT-based Chebyshev interpolation
    1. Compute Chebyshev coefficients via DCT
    2. Pad with zeros for higher frequencies
    3. Evaluate at fine grid points via inverse DCT
  - For pressure: Lagrange polynomial interpolation (order N-2)
- [ ] Wrap into `prolongate_fields(U_coarse, level_coarse, level_fine)` for (u, v, p)

### 1.4 Per-Level Solver Routines
- [ ] Refactor `_compute_residuals(level, u, v, p)` to use level's operators
- [ ] Refactor `_enforce_boundary_conditions(level, u, v)`
- [ ] Refactor `_compute_adaptive_timestep(level)` using level's dx_min, dy_min
- [ ] Create `smooth(level, n_steps)` - multiple RK4 pseudo-time steps on one level

### 1.5 FSG Driver
- [ ] Implement `solve_fsg(n_levels)`:
  ```
  for level in levels (coarse to fine):
      if level == coarsest:
          initialize with zeros
      else:
          prolongate solution from previous level

      while not converged:
          smooth(level, 1)  # RK4 steps
          check convergence on this level
  ```
- [ ] Use relaxed convergence criterion on coarse levels (e.g., 10x tolerance)
- [ ] Log per-level iteration counts and residuals

### 1.6 FSG Validation
- [ ] Compare FSG vs SG for N=32, 64 at Re=100
- [ ] Verify same solution accuracy
- [ ] Measure speedup (expect ~10-18x)

---

## Phase 2: VMG (V-cycle Multigrid with FAS)

### 2.1 Restriction Operator (Residuals)
- [ ] Implement `restrict_residual(r_fine, level_fine, level_coarse)`:
  - FFT-based: compute Chebyshev coefficients, truncate high frequencies
  - Row-column algorithm for 2D (restrict x-direction, then y-direction)
  - Set boundary residuals to zero (velocities have Dirichlet BCs)
- [ ] Handle inner-grid restriction for pressure residuals

### 2.2 Restriction Operator (Variables)
- [ ] Implement `restrict_solution(u_fine, level_fine, level_coarse)`:
  - Direct injection: coarse nodes are subset of fine nodes
  - `u_coarse[i,j] = u_fine[2i, 2j]`

### 2.3 Prolongation Operator (Corrections)
- [ ] Implement `prolongate_correction(c_coarse, level_coarse, level_fine)`:
  - Same FFT-based approach as solution prolongation
  - For pressure: Lagrange interpolation on inner grid

### 2.4 FAS Coarse-Grid Equation
- [ ] Compute τ-correction (FAS forcing term):
  ```
  τ = R(I_h^H * u_h) - I_h^H * R(u_h)
  ```
  where R is the residual operator, I_h^H is restriction
- [ ] Modified coarse-grid solve includes τ in RHS

### 2.5 V-cycle Implementation
- [ ] Implement `v_cycle(level)`:
  ```python
  def v_cycle(level):
      if level == coarsest:
          # Solve to convergence (or many smoothing steps)
          while not converged:
              smooth(level, 1)
          return

      # 1. Presmooth on fine level
      smooth(level, n_pre)  # typically n_pre = 1-3

      # 2. Compute fine-grid residual
      r_fine = compute_residual(level)

      # 3. Restrict solution and residual to coarse
      u_coarse_old = restrict_solution(u_fine, level, level-1)
      r_coarse = restrict_residual(r_fine, level, level-1)

      # 4. Set up FAS coarse problem (includes τ-correction)
      # 5. Recursively solve on coarse grid
      v_cycle(level - 1)

      # 6. Compute and prolongate correction
      correction = u_coarse_new - u_coarse_old
      correction_fine = prolongate_correction(correction, level-1, level)

      # 7. Update fine-grid solution
      u_fine += correction_fine

      # 8. Postsmooth (often unnecessary per Zhang & Xi)
      # smooth(level, n_post)
  ```

### 2.6 VMG Driver
- [ ] Implement `solve_vmg()`:
  ```
  initialize finest level with zeros
  while not converged:
      v_cycle(finest_level)
      check convergence
  ```

### 2.7 VMG Validation
- [ ] Compare VMG vs SG and FSG for N=32, 64 at Re=100
- [ ] Verify solution accuracy maintained
- [ ] Measure speedup (expect additional 50-90% over FSG)

---

## Phase 3: FMG (Full Multigrid)

### 3.1 FMG Driver
- [ ] Implement `solve_fmg()`:
  ```python
  def solve_fmg():
      # Start on coarsest level
      level = coarsest
      initialize_with_zeros(level)
      solve_to_convergence(level)  # SG on coarsest

      # Work up through levels
      for level in levels[1:]:  # coarse to fine
          # Prolongate converged solution as initial guess
          u_init = prolongate_solution(u_converged, level-1, level)

          # Apply V-cycles until convergence
          while not converged:
              v_cycle(level)  # Uses all levels from 0 to current
              check convergence

      return solution on finest level
  ```

### 3.2 FMG Validation
- [ ] Compare FMG vs all other methods
- [ ] Verify solution accuracy
- [ ] Measure speedup (expect ~100x over SG)

---

## Phase 4: Integration & Polish

### 4.1 Solver Interface
- [ ] Add `multigrid_method` parameter: `"none"`, `"fsg"`, `"vmg"`, `"fmg"`
- [ ] `solve()` dispatches to appropriate driver
- [ ] Backward compatible: `multigrid_method="none"` uses current SG implementation

### 4.2 Configuration
- [ ] Extend `SpectralParameters` with:
  ```python
  n_levels: int = 3
  multigrid_method: str = "fmg"
  n_presmooth: int = 1
  n_postsmooth: int = 0
  coarse_tolerance_factor: float = 10.0
  ```

### 4.3 Logging & Metrics
- [ ] Log per-level residuals
- [ ] Track V-cycle count
- [ ] Record time spent per level
- [ ] MLflow integration for multigrid metrics

### 4.4 Comprehensive Validation
- [ ] Ghia benchmark comparison for all methods
- [ ] Re = 100, 400, 1000
- [ ] N = 32, 64, 96
- [ ] Speedup plots (wall clock time vs N)
- [ ] Accuracy verification (solution differences < tolerance)

---

## Key Implementation Notes from Zhang & Xi (2010)

1. **Postsmoothing often unnecessary** - can cause instabilities
2. **Presmoothing steps**: 1-3 typically sufficient (paper uses 1,2,3 for levels)
3. **Convergence criterion**: RMS of continuity residual < ε (ε = 10⁻⁴ for cavity flow)
4. **CFL**: Keep around 2.5 (upper bound of RK4 stability)
5. **β² = 5.0**: Artificial compressibility coefficient (don't change)
6. **Corner singularities**: Your corner_smoothing handles this (paper uses subtraction method)

## Expected Results (Re=100, N=96)

| Method | Relative Time |
|--------|---------------|
| SG | 100% |
| FSG-3 | ~5.5% |
| VMG-111 | ~1.5% |
| FMG-111 | ~1.2% |
