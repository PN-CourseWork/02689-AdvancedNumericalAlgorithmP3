# Project 3: Lid-Driven Cavity Flow

Comparing Finite Volume and Spectral methods for the incompressible Navier-Stokes equations.

## Documentation

[Read the full documentation](https://02689-advancednumericalalgorithmproject3.readthedocs.io/en/latest/)

## Installation

```bash
uv sync
```

## Running Solvers

The project uses [Hydra](https://hydra.cc/) for configuration management. Run solvers via `run_solver.py`:

### Basic Usage

```bash
# Finite Volume solver (32x32 cells, Re=100)
uv run python run_solver.py solver=fv N=32 Re=100

# Spectral solver (N=15 gives 16x16 nodes, Re=100)
uv run python run_solver.py solver=spectral N=15 Re=100
```

### Using Experiment Configs

Pre-defined experiment configurations are in `conf/experiment/`:

```bash
# Quick test (small grid, few iterations)
uv run python run_solver.py +experiment=quick_test solver=fv

# FV validation (default settings for benchmarking)
uv run python run_solver.py +experiment=fv_validation

# Spectral validation
uv run python run_solver.py +experiment=spectral_validation
```

### Parameter Sweeps

Run multiple configurations with Hydra's multirun:

```bash
# Sweep over grid sizes
uv run python run_solver.py -m solver=fv N=16,32,64 Re=100

# Sweep over Reynolds numbers
uv run python run_solver.py -m solver=spectral N=31 Re=100,400,1000

# Full validation sweep
uv run python run_solver.py -m +experiment=fv_validation
```

### Configuration Structure

```
conf/
├── config.yaml              # Main config (N, Re, tolerance, etc.)
├── solver/
│   ├── fv.yaml              # FV-specific (alpha_uv, alpha_p, scheme)
│   └── spectral.yaml        # Spectral-specific (CFL, beta_squared)
├── experiment/
│   ├── quick_test.yaml      # Fast debugging runs
│   ├── fv_validation.yaml   # FV benchmark settings
│   └── spectral_validation.yaml
└── mlflow/
    ├── local.yaml           # Local file-based tracking
    └── databricks.yaml      # Cloud tracking
```

### Output Structure

Each run creates a timestamped output directory:

```
outputs/{experiment_name}/Re{Re}/{solver}/{DD-MM-YY}/{HH:MM:SS}/
├── config.yaml              # Resolved configuration
├── run_solver.log           # Execution log
├── LDC_{SOLVER}_N{N}_Re{Re}.h5  # Results (HDF5)
└── .hydra/                  # Hydra internals
```

## Viewing Results

### MLflow UI

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 5001
```

Then open http://localhost:5001

### Logged Data

- **Parameters:** solver settings, grid size, Reynolds number
- **Metrics:** iterations, convergence, wall time, residuals
- **Timeseries:** residual history, energy, enstrophy (with step)
- **Artifacts:** HDF5 result files

## References

- [High-Re solutions for incompressible flow (Ghia et al.)](https://www.sciencedirect.com/science/article/pii/0021999182900584) - Benchmark data
- [Chebyshev pseudospectral multigrid method](https://www.sciencedirect.com/science/article/pii/S0045793009001121) - Spectral method
- [The 2D lid-driven cavity problem revisited](https://www.researchgate.net/publication/222433759_The_2D_lid-driven_cavity_problem_revisited) - Conserved quantities
- [P_N-P_{N-2} spectral method](https://www.sciencedirect.com/science/article/pii/S0743731518305549) - Pressure formulation
