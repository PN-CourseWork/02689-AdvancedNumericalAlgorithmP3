# Hyperparameter Optimization with Optuna

Optimize the `corner_smoothing` parameter for the spectral solver using Optuna + Hydra.

## Quick Start

```bash
# FV L2 error (default)
uv run python main.py -m +experiment/optimization=corner_smoothing \
    'solver.corner_smoothing=interval(0.02,0.35)'

# Botella vortex metrics
uv run python main.py -m +experiment/optimization=corner_smoothing \
    'solver.corner_smoothing=interval(0.02,0.35)' optuna.objective=botella_vortex
```

## Objectives

| Objective | Description |
|-----------|-------------|
| `fv_l2_error` (default) | Minimize L2 error vs FV reference (true LDC) |
| `botella_vortex` | Minimize vortex metric error vs Botella & Peyret |

Note: Multi-objective Pareto optimization is not supported by `hydra-optuna-sweeper` 1.x (required for Optuna 2.x compatibility).

## Configuration Options

```bash
# Change Reynolds number
... Re=1000

# Change grid size
... N=32

# Change number of trials
... hydra.sweeper.n_trials=30

# Change parallel jobs
... hydra.sweeper.n_jobs=8

# Change search range
'solver.corner_smoothing=interval(0.01,0.5)'
```

## Defaults

| Parameter | Value |
|-----------|-------|
| Solver | FSG (spectral multigrid) |
| N | 24 |
| Re | 400 |
| Tolerance | 1e-6 |
| Trials | 10 |
| Parallel jobs | 4 |

## Results

Logged to MLflow under `Optuna-CornerSmoothing-{objective}`. View with:

```bash
uv run mlflow ui
```

## Config Files

| File | Purpose |
|------|---------|
| `conf/experiment/optimization/corner_smoothing.yaml` | Optimization experiment |
| `conf/hydra/sweeper/optuna_corner.yaml` | Optuna sweeper settings |
