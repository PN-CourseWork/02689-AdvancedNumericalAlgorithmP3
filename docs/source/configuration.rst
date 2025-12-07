Configuration
=============

This guide explains the Hydra configuration system used for experiment management.

Configuration Structure
-----------------------

.. code-block:: text

   conf/
   ├── config.yaml              # Main config (defaults, grid size, tolerance)
   ├── problem/
   │   └── ldc.yaml             # Physics (Re, lid velocity, domain size)
   ├── solver/
   │   ├── fv.yaml              # Finite Volume solver
   │   └── spectral/            # Spectral solver variants
   │       ├── sg.yaml          # Single Grid
   │       ├── fsg.yaml         # Full Single Grid MG
   │       ├── vmg.yaml         # V-cycle MultiGrid
   │       └── fmg.yaml         # Full MultiGrid
   ├── experiment/
   │   ├── validation/ghia/     # Ghia benchmark validation
   │   │   ├── fv.yaml
   │   │   └── spectral.yaml
   │   └── benchmarking/
   │       └── timings.yaml
   ├── machine/
   │   ├── local.yaml           # Local machine settings
   │   └── hpc.yaml             # DTU HPC cluster
   └── mlflow/
       ├── local.yaml           # File-based tracking (default)
       └── coolify.yaml         # Remote server

Base Configuration
------------------

The main ``config.yaml`` defines defaults and grid parameters:

.. code-block:: yaml

   # conf/config.yaml
   defaults:
     - problem: ldc
     - solver: fv
     - mlflow: local
     - machine: local
     - _self_

   N: 32                    # Grid size
   tolerance: 1.0e-6        # Convergence tolerance
   max_iterations: 10000    # Maximum iterations

   experiment_name: LDC-Dev
   sweep_name: dev-run
   plot_only: false         # Set true to regenerate plots without solving

Problem Configuration
---------------------

Physics parameters are defined in ``conf/problem/ldc.yaml``:

.. code-block:: yaml

   # @package _global_
   Re: 100              # Reynolds number
   lid_velocity: 1.0    # Velocity of moving lid
   Lx: 1.0              # Domain width
   Ly: 1.0              # Domain height

Solver Configurations
---------------------

Solvers use Hydra interpolation (``${...}``) to inherit parameters from the root config.

**Finite Volume** (``conf/solver/fv.yaml``):

.. code-block:: yaml

   # @package solver
   _target_: solvers.fv.solver.FVSolver
   name: fv

   # Interpolated from root config
   Re: ${Re}
   nx: ${N}
   ny: ${N}
   tolerance: ${tolerance}

   # FV-specific parameters
   convection_scheme: TVD    # TVD or upwind
   limiter: MUSCL            # MUSCL, minmod, vanLeer
   alpha_uv: 0.6             # Velocity under-relaxation
   alpha_p: 0.4              # Pressure under-relaxation

**Spectral Single Grid** (``conf/solver/spectral/sg.yaml``):

.. code-block:: yaml

   # @package solver
   _target_: solvers.spectral.sg.SGSolver
   name: spectral

   Re: ${Re}
   nx: ${N}
   ny: ${N}

   # Spectral-specific parameters
   basis_type: chebyshev     # chebyshev or legendre
   CFL: 0.5
   beta_squared: 5.0         # Artificial compressibility

   # Corner singularity treatment
   corner_treatment: smoothing   # smoothing or subtraction
   corner_smoothing: 0.15

Other spectral variants (``fsg.yaml``, ``vmg.yaml``, ``fmg.yaml``) add multigrid acceleration.

Experiment Configurations
-------------------------

Experiments override base settings and define parameter sweeps.

**Ghia Validation** (``conf/experiment/validation/ghia/fv.yaml``):

.. code-block:: yaml

   # @package _global_
   defaults:
     - override /solver: fv

   experiment_name: LDC-Validation
   sweep_name: ghia-Re${Re}

   hydra:
     sweeper:
       params:
         N: 32, 64
         Re: 100

Command Line Usage
------------------

Single Runs
^^^^^^^^^^^

.. code-block:: bash

   # Default FV solver
   uv run python main.py solver=fv N=32 Re=100

   # Spectral solver (single grid)
   uv run python main.py solver=spectral/sg N=31 Re=100

   # Spectral with V-cycle multigrid
   uv run python main.py solver=spectral/vmg N=31 Re=1000

Parameter Sweeps
^^^^^^^^^^^^^^^^

Use ``-m`` (multirun) flag:

.. code-block:: bash

   # Command-line sweep
   uv run python main.py -m solver=fv N=16,32,64 Re=100,400

   # Use experiment config
   uv run python main.py -m +experiment/validation/ghia=fv

   # Override experiment parameters
   uv run python main.py -m +experiment/validation/ghia=fv Re=400,1000

Plot-Only Mode
^^^^^^^^^^^^^^

Regenerate plots from existing MLflow runs without re-solving:

.. code-block:: bash

   uv run python main.py -m +experiment/validation/ghia=fv plot_only=true

Viewing Configuration
^^^^^^^^^^^^^^^^^^^^^

Print resolved config without running:

.. code-block:: bash

   uv run python main.py --cfg job
   uv run python main.py +experiment/validation/ghia=fv --cfg job

MLflow Integration
------------------

Runs are automatically tracked in MLflow. The ``experiment_name`` determines the
MLflow experiment, and ``sweep_name`` creates parent runs for grouping sweeps:

.. code-block:: text

   LDC-Validation (experiment)
   └── ghia-Re100 (parent run)
       ├── fv_N32 (child)
       └── fv_N64 (child)

View results:

.. code-block:: bash

   uv run mlflow ui
   # Open http://localhost:5000
