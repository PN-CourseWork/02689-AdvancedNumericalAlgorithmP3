# Project 3: Lid-Driven Cavity Flow

## Documentation

📚 [Read the full documentation](https://02689-advancednumericalalgorithmproject3.readthedocs.io/en/latest/)

## Installation

Run the setup script from project root:
```bash
bash setup.sh
```

## HPC (LSF Cluster)

Submit parameter sweeps using job packs:

```bash
# Submit spectral solver jobs
uv run python main.py --hpc spectral

# Submit FV solver jobs
uv run python main.py --hpc fv

# Submit all jobs
uv run python main.py --hpc all
```

Edit `Experiments/*/generate_pack.sh` to customize resources and parameter sweep values.

### Monitoring and Managing Jobs

```bash
# Check job status
bstat

# Kill a specific job by name
bkill -J Spectral-N23-Re100

# Kill a job by ID
bkill 27198795

# Kill all your jobs
bkill 0
```

## References


### SIMPLE for Spectral
[A spectral pressure correction method for unsteady incompressible flows](https://www.sciencedirect.com/science/article/pii/S0021999112007334)

### Multigrid Method
[An explicit Chebyshev pseudospectral multigrid method for incompressible Navier–Stokes equations](https://www.sciencedirect.com/science/article/pii/S0045793009001121)

### Quantities
[The 2D lid-driven cavity problem revisited](https://www.researchgate.net/publication/222433759_The_2D_lid-driven_cavity_problem_revisited)

### Ghia Benchmark
[High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method](https://www.sciencedirect.com/science/article/pii/0021999182900584)

### P_N - P_{N-2} Method
[Parallel spectral-element direction splitting method for incompressible Navier–Stokes equations](https://www.sciencedirect.com/science/article/pii/S0743731518305549)
