# Implementation Plan: Solution Export Pipeline

## Architecture Overview

```
SOLVERS (FV, Spectral, Multigrid)
    │
    ├── to_vtk() → solution.vts (u, v, p, vorticity, velocity_mag, metadata)
    ├── compute_global_quantities() → E, Z, P
    └── [Spectral] save_coefficients() → coefficients.npz
    │
    ▼
MLflow Artifacts
    ├── fields/solution.vts
    ├── timeseries/history.parquet
    ├── model/coefficients.npz  [spectral only]
    └── config.yaml
    │
    ▼
PLOTTING (solver-agnostic)
    load_solution() → pv.StructuredGrid → all plot functions
```

---

## Implementation Steps

### 1. Solver Base Class
Add to `src/solvers/base.py`:
```python
@abstractmethod
def to_vtk(self) -> pv.StructuredGrid:
    """Export solution with fields + metadata."""

@abstractmethod
def compute_global_quantities(self) -> dict[str, float]:
    """Return {'E': ..., 'Z': ..., 'P': ...}"""
```

### 2. FV Solver
- `to_vtk()`: Create grid, add u/v/p, compute vorticity with FV gradients
- `compute_global_quantities()`: Integrate using cell volumes

### 3. Spectral Solver
- `to_vtk()`: Create grid, add u/v/p, compute vorticity with spectral differentiation
- `compute_global_quantities()`: Integrate using Gauss-Lobatto quadrature weights
- `save_coefficients()`: Export u_hat, v_hat, p_hat as .npz

### 4. Multigrid Solver
- Delegate to underlying spectral solver

### 5. run_solver.py
Replace `log_fields()` with:
```python
def log_solution(solver):
    grid = solver.to_vtk()
    # Save VTS with compression
    # Save timeseries as Parquet
    # Save coefficients if spectral
    # Log E, Z, P as metrics
```

### 6. Plotting Pipeline
- Rename `load_fields_from_zarr()` → `load_solution()`
- Remove reshaping hacks (VTS already structured)
- Add `load_coefficients()` utility for Ghia interpolation

### 7. Cleanup
- Remove zarr dependencies
- Remove backwards compatibility code

---

## VTS Contents

| Field | Source |
|-------|--------|
| u, v, p | Primary fields |
| velocity_magnitude | `sqrt(u² + v²)` |
| vorticity | `∂v/∂x - ∂u/∂y` (native differentiation) |
| velocity (vector) | `[u, v, 0]` |
| Re, N, solver | Metadata (field_data) |

---

## Global Quantities

| Quantity | Formula | Method |
|----------|---------|--------|
| E (Kinetic Energy) | `½ ∫ \|u\|² dΩ` | Quadrature |
| Z (Enstrophy) | `½ ∫ ω² dΩ` | Quadrature |
| P (Palinstrophy) | `½ ∫ \|∇ω\|² dΩ` | Native grad + quadrature |
