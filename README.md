# Project 3: Lid-Driven Cavity Flow

Comparing Finite Volume and Spectral methods for the incompressible Navier-Stokes equations.

## Documentation

[Read the full documentation](https://02689-advancednumericalalgorithmproject3.readthedocs.io/en/latest/)

## Installation

```bash
uv sync
```

## Usage

```bash
# Run solver + generate plots (default)
uv run python main.py -m +experiment/validation/ghia=fv

# Regenerate plots only (no solving)
uv run python main.py -m +experiment/validation/ghia=fv plot_only=true

# Single run (testing)
uv run python main.py solver=fv N=32 Re=100

# Custom sweeps
uv run python main.py -m solver=fv N=16,32,64 Re=100,400
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration. Structure:

```
conf/
├── config.yaml              # Main config (N, Re, tolerance)
├── solver/
│   ├── fv.yaml              # Finite Volume settings
│   └── spectral/            # Spectral solver variants
├── experiment/
│   └── validation/ghia/     # Ghia benchmark experiments
└── mlflow/
    ├── local.yaml           # File-based tracking (default)
    └── coolify.yaml         # Remote server
```

## MLflow

Results are tracked with [MLflow](https://mlflow.org/):

```bash
# Local UI
uv run mlflow ui

# Remote server
# https://kni.dk/mlflow-ana-p3/
```

## Hyperparameter Optimization

Optimize `corner_smoothing` using [Optuna](https://optuna.org/) + Hydra:

```bash
# Minimize L2 error vs FV reference (default objective)
uv run python main.py -m +experiment/optimization=corner_smoothing \
    'solver.corner_smoothing=interval(0.02,0.35)' Re=1000 N=30

# Minimize vortex error vs Botella & Peyret reference
uv run python main.py -m +experiment/optimization=corner_smoothing \
    'solver.corner_smoothing=interval(0.02,0.35)' Re=1000 N=30 \
    optuna.objective=botella_vortex

# Customize trials and parallelism
uv run python main.py -m +experiment/optimization=corner_smoothing \
    'solver.corner_smoothing=interval(0.02,0.35)' Re=1000 N=30 \
    hydra.sweeper.n_trials=20 hydra.sweeper.n_jobs=8
```

View results in MLflow under `Optuna-CornerSmoothing-{objective}`. See [docs/optuna_optimization.md](docs/optuna_optimization.md) for details.

## References

- [High-Re solutions for incompressible flow (Ghia et al.)](https://www.sciencedirect.com/science/article/pii/0021999182900584)
- [Chebyshev pseudospectral multigrid method](https://www.sciencedirect.com/science/article/pii/S0045793009001121)
- [The 2D lid-driven cavity problem revisited](https://www.researchgate.net/publication/222433759_The_2D_lid-driven_cavity_problem_revisited)
