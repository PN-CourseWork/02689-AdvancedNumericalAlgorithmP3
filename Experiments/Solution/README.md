# Solution Plotting Experiment

Generate publication-quality plots from MLflow solver runs. Uses the **same config** as `run_solver.py` to automatically find matching runs.

## Usage

```bash
# Run solver first
uv run python run_solver.py solver=spectral N=31 Re=100

# Plot with same config - finds matching run automatically!
uv run python Experiments/Solution/plot_solution.py solver=spectral N=31 Re=100

# With Ghia benchmark validation
uv run python Experiments/Solution/plot_solution.py solver=spectral N=31 Re=100 plots.ghia_validation=true

# FV solver example
uv run python Experiments/Solution/plot_solution.py solver=fv N=32 Re=400

# Use remote MLflow
uv run python Experiments/Solution/plot_solution.py solver=spectral N=31 Re=100 mlflow=coolify

# Disable upload back to MLflow
uv run python Experiments/Solution/plot_solution.py solver=fv N=64 Re=1000 upload_to_mlflow=false
```

## How It Works

1. Takes same Hydra config as `run_solver.py`
2. Queries MLflow for runs matching: `solver`, `N`, `Re`
3. Downloads solution artifacts (zarr fields)
4. Generates plots
5. Uploads plots back to the same MLflow run

## Generated Plots

| Plot | Description | Enable |
|------|-------------|--------|
| `fields.png` | Pressure, U, V velocity contours | `plots.fields=true` (default) |
| `streamlines.png` | Velocity magnitude with streamlines | `plots.streamlines=true` (default) |
| `vorticity.png` | Vorticity contour | `plots.vorticity=true` (default) |
| `centerlines.png` | U(y) and V(x) profiles | `plots.centerlines=true` (default) |
| `ghia_validation.png` | Ghia et al. (1982) comparison | `plots.ghia_validation=true` |

## Ghia Benchmark

Available Reynolds numbers: `Re ∈ {100, 400, 1000, 3200, 5000, 7500, 10000}`

## Output

- Plots saved to Hydra output directory
- Uploaded to MLflow run under `artifacts/plots/`
