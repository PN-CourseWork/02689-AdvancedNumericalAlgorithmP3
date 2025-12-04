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
    ├── local-files.yaml     # File-based tracking (default)
    ├── local-docker.yaml    # Local Docker server
    └── coolify.yaml         # Remote server (Coolify)
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

## MLflow

Results are tracked with [MLflow](https://mlflow.org/). Three tracking modes are available:

### Local Files (Default)

File-based tracking in `./mlruns` - no setup required:

```bash
uv run python run_solver.py solver=fv mlflow=local-files

# View UI
uv run mlflow ui --backend-store-uri ./mlruns --port 5001
```

### Local Docker

Run the official MLflow server in Docker:

```bash
# Start server
cd mlflow-server && docker compose up -d

# Run solver
uv run python run_solver.py solver=fv mlflow=local-docker

# View UI at http://localhost:5001
```

### Remote Server (Coolify)

Deploy `mlflow-server/docker-compose.yml` to Coolify, then:

```bash
export MLFLOW_TRACKING_URI=https://mlflow.yourdomain.com
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=yourpassword
uv run python run_solver.py solver=fv mlflow=coolify
```

**Authentication:** The server uses MLflow's built-in basic auth. Default credentials are `admin` / `password1234`. Change the password after first login via the UI or API:

```bash
curl -X PATCH -u admin:password1234 \
  "$MLFLOW_TRACKING_URI/api/2.0/mlflow/users/update-password" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "new-secure-password"}'
```

### Logged Data

- **Parameters:** solver settings, grid size, Reynolds number
- **Metrics:** iterations, convergence, wall time, residuals
- **Timeseries:** residual history, energy, enstrophy (per iteration)
- **Artifacts:** HDF5 result files

## References

- [High-Re solutions for incompressible flow (Ghia et al.)](https://www.sciencedirect.com/science/article/pii/0021999182900584) - Benchmark data
- [Chebyshev pseudospectral multigrid method](https://www.sciencedirect.com/science/article/pii/S0045793009001121) - Spectral method
- [The 2D lid-driven cavity problem revisited](https://www.researchgate.net/publication/222433759_The_2D_lid-driven_cavity_problem_revisited) - Conserved quantities
- [P_N-P_{N-2} spectral method](https://www.sciencedirect.com/science/article/pii/S0743731518305549) - Pressure formulation
