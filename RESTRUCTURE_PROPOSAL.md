# Proposed Project Restructure

## Goals

1. Mirror `src/` and `conf/` structure for readability
2. Introduce a `utilities/` submodule inside `src/` for cross-project helpers (MLflow/Hydra/HPC, logging, generic plotting/config)
3. Organize project-specific logic under `solvers/` and `shared/`
4. Spectral multigrid configs extend the base spectral config
5. Clean separation of concerns and discovery

## Proposed Structure

### Source Code (`src/`)

```
src/
├── __init__.py
├── solvers/
│   ├── __init__.py
│   ├── base.py                    # BaseSolver ABC
│   ├── metrics.py                 # Norms/formatting shared across solvers
│   ├── datastructures.py          # Solver data classes
│   ├── fv/
│   │   ├── __init__.py
│   │   ├── solver.py              # FVSolver class
│   │   ├── assembly/
│   │   │   ├── __init__.py
│   │   │   ├── convection_diffusion_matrix.py
│   │   │   ├── pressure_correction_eq_assembly.py
│   │   │   ├── divergence.py
│   │   │   └── rhie_chow.py
│   │   ├── discretization/
│   │   │   ├── __init__.py
│   │   │   ├── convection/
│   │   │   ├── diffusion/
│   │   │   └── gradient/
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── corrections.py
│   │   │   └── helpers.py
│   │   └── linear_solvers/
│   │       ├── __init__.py
│   │       └── scipy_solver.py
│   │
│   └── spectral/
│       ├── __init__.py
│       ├── solver.py              # SpectralSolver class (single grid)
│       ├── basis/
│       │   ├── __init__.py
│       │   ├── polynomial.py      # Jacobi, Legendre, interpolation
│       │   └── spectral.py        # Basis classes, diff matrices
│       ├── operators/
│       │   ├── __init__.py
│       │   ├── prolongation.py    # Prolongation operators
│       │   ├── restriction.py     # Restriction operators
│       │   └── corner.py          # Corner singularity treatment
│       ├── multigrid/
│       │   ├── __init__.py
│       │   ├── fsg.py             # Full Spectral Galerkin
│       │   ├── vmg.py             # V-cycle multigrid
│       │   └── fmg.py             # Full multigrid
│
├── utilities/                     # Cross-project utilities (reusable across repos)
│   ├── __init__.py
│   ├── config/                    # Config helpers (paths, cleanup) — domain-agnostic
│   ├── mlflow/
│   │   ├── __init__.py
│   │   ├── tracking.py
│   │   └── callback.py
│   ├── hydra/
│   │   ├── __init__.py
│   │   └── callbacks.py
│   ├── hpc/
│   │   ├── __init__.py
│   │   └── sweeper.py
│   ├── plotting/                  # Generic plotting style/format helpers
│   │   ├── __init__.py
│   │   ├── styles.py
│   │   └── formatters.py
│   ├── runners/                   # Generic CLI helpers (if kept)
│   ├── io.py                      # Generic parquet/pickle helpers (from spectral/utils/io.py)
│   └── data_io.py                 # If broadly reusable; else move to shared/
│
├── shared/                       # Project-scoped shared code (specific to this LDC project)
│   ├── __init__.py
│   ├── meshing/                   # Domain meshes specific to this project
│   │   ├── __init__.py
│   │   ├── mesh_data.py
│   │   └── structured.py
│   ├── plotting/                  # Project-specific plots (Ghia, LDC visuals)
│   │   ├── __init__.py
│   │   ├── solution.py            # Solution visualization
│   │   ├── validation.py          # Ghia comparison plots
│   │   ├── convergence.py         # Convergence history plots
│   │   ├── ldc_plotter.py         # From utils/ldc_plotter.py
│   │   ├── styles.py              # Plot styling utilities (incl. mplstyle)
│   │   └── scientific.mplstyle
│   ├── data_io.py                 # Project-scoped data I/O (LDC-specific shapes/conventions)
│   ├── field_interpolator.py      # Interpolation tailored to project grids
│   ├── ghia_validator.py          # Project-specific validation
│   └── datastructures.py          # Project-scoped data classes (if any)
│
```

### Configuration (`conf/`)

```
conf/
├── config.yaml                    # Main config with defaults
│
├── solvers/                       # Mirrors src/solvers/ hierarchy
│   ├── fv/                        # Group: solvers/fv
│   │   ├── default.yaml           # FV solver parameters (baseline)
│   │   └── linear_solvers/
│   │       ├── default.yaml       # Default linear solver selection
│   │       └── scipy.yaml         # SciPy backend settings
│   │
│   └── spectral/                  # Group: solvers/spectral
│       ├── default.yaml           # Base spectral (single-grid)
│       ├── basis/
│       │   └── default.yaml       # Basis selection (legendre/chebyshev, quad rules)
│       ├── operators/
│       │   ├── default.yaml       # Corner treatment defaults (smoothing width, method)
│       │   ├── transfer.yaml      # Prolongation/restriction choices
│       │   └── corner.yaml        # Alternate corner options if needed
│       ├── multigrid/
│       │   ├── default.yaml       # Placeholder (disabled multigrid)
│       │   ├── fsg.yaml           # Full Spectral Galerkin
│       │   ├── vmg.yaml           # V-cycle multigrid
│       │   └── fmg.yaml           # Full multigrid
│       └── metrics/
│           └── default.yaml       # Norm/formatting knobs (optional group)
│
├── shared/                        # Cross-project infra configs
│   ├── mlflow/
│   │   ├── default.yaml           # Local MLflow as baseline
│   │   └── coolify.yaml
│   └── hydra/
│       └── launcher/
│           ├── default.yaml       # Local/single-run baseline
│           └── joblib.yaml
│
└── experiment/
    ├── fv_validation.yaml         # FV grid convergence study
    ├── spectral_validation.yaml   # Spectral grid convergence
    ├── solver_comparison.yaml     # FV vs spectral baseline
    ├── corner_treatment.yaml      # Corner singularity comparison
    ├── multigrid_comparison.yaml  # SG vs FSG vs VMG vs FMG
    └── transfer_operators.yaml    # Prolongation/restriction comparison
```

## Configuration Details

### Main Config (`conf/config.yaml`)

```yaml
defaults:
  - solvers/spectral                # Uses solvers/spectral/default.yaml
  - solvers/spectral/operators      # Uses solvers/spectral/operators/default.yaml
  - solvers/spectral/operators@transfer: transfer  # Pick transfer variant
  - solvers/spectral/basis          # Uses solvers/spectral/basis/default.yaml
  - solvers/spectral/multigrid: default            # Multigrid disabled by default
  - shared/mlflow: default          # Local MLflow baseline
  - shared/hydra/launcher: default  # Local launcher baseline
  - _self_

# Problem parameters
Re: 100
N: 16
Lx: 1.0
Ly: 1.0
lid_velocity: 1.0

# Solver parameters
max_iterations: 10000
tolerance: 1.0e-6
```

### Base Spectral Config (`conf/solvers/spectral/default.yaml`)

```yaml
# @package _global_
solver: spectral

# Spectral-specific parameters
basis_type: legendre
CFL: 0.5
corner_treatment: smoothing

# Single-grid (no multigrid)
multigrid:
  enabled: false
```

### FSG Multigrid Config (`conf/solvers/spectral/fsg.yaml`)

```yaml
# @package _global_
defaults:
  - default  # Inherit from base spectral

solver: spectral_fsg

multigrid:
  enabled: true
  type: fsg
  coarse_N: 4
  prolongation: polynomial
  restriction: injection
```

### VMG Config (`conf/solvers/spectral/vmg.yaml`)

```yaml
# @package _global_
defaults:
  - default

solver: spectral_vmg

multigrid:
  enabled: true
  type: vmg
  levels: 3
  smoothing_steps: 2
  prolongation: fft
  restriction: fft
```

### FV Config (`conf/solvers/fv/default.yaml`)

```yaml
# @package _global_
solver: fv

# FV-specific parameters
convection_scheme: upwind
limiter: none
pressure_solver: scipy
relaxation:
  velocity: 0.7
  pressure: 0.3
```

### Example Experiment (`conf/experiment/multigrid_comparison.yaml`)

```yaml
# @package _global_
defaults:
  - override /solvers/spectral: default

experiment_name: Multigrid-Comparison
sweep_name: sg_vs_fsg_vs_vmg

Re: 100
tolerance: 1.0e-6
max_iterations: 50000

hydra:
  sweeper:
    params:
      solvers/spectral: default,fsg,vmg
      N: 16,32,64
```

## Migration Plan

### Phase 1: Reorganize `src/`

1. Create new directory structure
2. Move files to new locations:
   - `src/solvers/*.py` → `src/solvers/base.py`
   - `src/solvers/fv_solver.py` → `src/solvers/fv/solver.py`
   - `src/solvers/spectral_solver.py` → `src/solvers/spectral/solver.py`
   - `src/solvers/datastructures.py` → `src/solvers/datastructures.py` (kept beside base)
   - `src/fv/*` → `src/solvers/fv/*`
   - `src/spectral/spectral.py` → `src/solvers/spectral/solver.py`
   - `src/spectral/polynomial.py` → `src/solvers/spectral/basis/polynomial.py`
   - `src/spectral/corner_singularity.py` → `src/solvers/spectral/operators/corner.py`
   - `src/spectral/transfer_operators.py` → split into `src/solvers/spectral/operators/{prolongation.py,restriction.py}`
   - `src/spectral/utils/formatting.py`, `norms.py` → `src/solvers/metrics.py` (shared across FV & spectral)
   - `src/spectral/utils/io.py` → `src/utilities/io.py` (generic parquet/pickle loader)
   - `src/spectral/utils/plotting.py` → `src/shared/plotting/` (project-level plotting helpers)
   - `src/spectral/spectral_multigrid.py` → split into `src/solvers/spectral/multigrid/{fsg.py,vmg.py,fmg.py}`
   - Remove `src/fv/linear_solvers/petsc_solver.py` and any references (we keep SciPy backend only)
   - `src/meshing/*` → `src/shared/meshing/*`
   - `src/plotting.py` + `src/utils/plotting/*` → `src/shared/plotting/` (project-level plotting)
   - `src/utils/data_io.py` → `src/shared/data_io.py` (or `utilities/` if made fully generic)
   - `src/utils/field_interpolator.py` → `src/shared/field_interpolator.py`
   - `src/utils/ghia_validator.py` → `src/shared/ghia_validator.py`
   - `src/utils/ldc_plotter.py` → `src/shared/plotting/ldc_plotter.py`
   - `src/utils/config/*` → `src/utilities/config/*`
   - `src/utils/mlflow/*` → `src/utilities/mlflow/*`
   - `src/utils/mlflow_callback.py` → `src/utilities/mlflow/callback.py`
   - `src/utils/hydra/*` → `src/utilities/hydra/*`
   - `src/utils/hpc/*` → `src/utilities/hpc/*`
   - `src/utils/runners/*` (if project-agnostic) → `src/utilities/`
   - `src/utils/mlflow_io.py`, `upload_logs.py`, `logs.py` → `src/utilities/mlflow/`
   - `src/utils/*` that are project-specific → keep or move into `shared/` as appropriate
3. Update all imports
4. Update `__init__.py` exports
5. Delete stale `__pycache__/` and `*.nbc` artifacts after moves to avoid packaging noise

### Phase 2: Reorganize `conf/`

1. Create new directory structure
2. Move configs:
   - `conf/solver/fv.yaml` → `conf/solvers/fv/default.yaml`
   - `conf/solver/spectral.yaml` → `conf/solvers/spectral/default.yaml`
   - `conf/mlflow/*` → `conf/shared/mlflow/*` (with `default.yaml`)
   - `conf/hydra/*` → `conf/shared/hydra/*` (with `launcher/default.yaml`)
   - `conf/experiment/multigrid.yaml` → `conf/experiment/multigrid_comparison.yaml`
   - Keep `solver_comparison.yaml`, `transfer_operators.yaml`, `spectral_validation.yaml`, `fv_validation.yaml`, `corner_treatment.yaml`
3. Create multigrid variant configs (fsg.yaml, vmg.yaml, fmg.yaml)
4. Update `config.yaml` defaults
5. Update experiment configs

### Phase 3: Update `run_solver.py`

1. Simplify solver instantiation using Hydra's `instantiate()`
2. Remove manual parameter extraction
3. Use `_target_` pattern for automatic class instantiation

## Benefits

1. **Discoverability**: Mirror structure makes it easy to find config for any code
2. **Extensibility**: Adding new solver = add directory in both `src/solvers/` and `conf/solvers/`
3. **Inheritance**: Multigrid configs extend base spectral, reducing duplication
4. **Clean sweeps**: Each solver type has appropriate N ranges in its experiments
5. **Separation**: Infrastructure code separated from numerical methods
