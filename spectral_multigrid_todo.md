
# Spectral Multigrid TODO List

## 1. Refactor for Multi-Level Support
- [ ] Introduce a `SpectralLevel` class holding:
  - basis objects
  - grids (full + inner)
  - shapes
  - differentiation matrices
  - spectral field arrays
- [ ] Refactor current single-level routines to take a `level` argument:
  - `_setup_grids(level)`
  - `_build_diff_matrices(level)`
  - `_initialize_lid_velocity(level)`
  - `_compute_residuals(level, u, v, p)`
  - `_enforce_boundary_conditions(level, u, v)`
  - `_compute_adaptive_timestep(level)`

## 2. Build the Grid Hierarchy
- [ ] Define number of levels and `(nx, ny)` per level.
- [ ] Ensure coarse Lobatto nodes are subsets of fine nodes.
- [ ] For each level, construct basis, grids, operators, fields, BCs.

## 3. Restriction & Prolongation
- [ ] Build restriction operators for:
  - u, v on full grids
  - p on inner grids
- [ ] Build prolongation operators for:
  - u, v on full grids
  - p on inner grids
- [ ] Wrap them into helper functions for U = (u, v, p).

## 4. Turn RK4 Into a Per-Level Smoother
- [ ] Generalize `step()` to operate on a specific level.
- [ ] Create `smooth(level, n_steps)` calling multiple RK4 pseudo-steps.

## 5. Residuals Per Level
- [ ] Implement `compute_residual_norms(level)` using that level's arrays.
- [ ] Ensure `_compute_residuals` operates correctly per level.

## 6. Full Approximation Scheme (FAS) V-cycle
- [ ] Presmoothing on level ℓ.
- [ ] Compute fine-grid residuals.
- [ ] Restrict U and F(U) to coarse grid.
- [ ] Compute FAS τ term for coarse grid.
- [ ] Recursive coarse solve via `v_cycle(ℓ-1)`.
- [ ] Prolongate coarse correction.
- [ ] Apply correction to fine-level U.
- [ ] Postsmoothing.

## 7. Coarsest Level Solver
- [ ] Implement `v_cycle(0)` as many smoothing iterations until convergence.

## 8. Full Multigrid (FMG) Driver
- [ ] Coarse-level initialization and solve.
- [ ] Prolongate to next level and apply V-cycles.
- [ ] Continue until finest level reached.

## 9. Integration with Existing Solver Interface
- [ ] `solve()` calls FMG or repeated V-cycles.
- [ ] Logging residuals from finest level.
- [ ] `_finalize_fields()` pulls finest-level fields into global output.

## 10. Parameters & Tuning
- [ ] Extend `SpectralParameters` with multigrid parameters.
- [ ] Log per-cycle residuals, work units, timings.

