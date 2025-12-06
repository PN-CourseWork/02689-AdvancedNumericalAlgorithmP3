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

### Using Experiment Configs

Pre-defined experiment configurations are in `conf/experiment/`:

```bash
uv run python run_solver.py -m +experiment=fv_validation
```

for only plots: pass  plot_only=true

Overwriting at runtime: 
uv run python run_solver.py -m +experiment=fv_validation N=16,32,64 Re=100


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
├── mlflow/
│   ├── local.yaml           # File-based tracking (default)
│   └── coolify.yaml         # Remote server (Coolify)
└── hydra/
    └── launcher/
        └── joblib.yaml      # Parallel launcher (all cores)
```

## MLflow

Results are tracked with [MLflow](https://mlflow.org/). Two tracking modes are available:

### Local Files (Default)

File-based tracking in `./mlruns` - no setup required:

```bash
uv run python run_solver.py solver=fv mlflow=local

# View UI
uv run main.py --mlflow-ui
```

### Remote Server (Coolify)

[mlflow-server](https://kni.dk/mlflow-ana-p3/#/experiments) 
```bash
# Setup credentials (one-time)
cp .env.template .env
# Edit .env with your credentials

# Run solver
uv run python run_solver.py solver=fv mlflow=coolify
```


## References

- [High-Re solutions for incompressible flow (Ghia et al.)](https://www.sciencedirect.com/science/article/pii/0021999182900584) - Benchmark data
- [Chebyshev pseudospectral multigrid method](https://www.sciencedirect.com/science/article/pii/S0045793009001121) - Spectral method
- [The 2D lid-driven cavity problem revisited](https://www.researchgate.net/publication/222433759_The_2D_lid-driven_cavity_problem_revisited) - Conserved quantities
- [P_N-P_{N-2} spectral method](https://www.sciencedirect.com/science/article/pii/S0743731518305549) - Pressure formulation
