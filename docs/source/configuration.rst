Experiment Configuration
========================

This guide explains the Hydra configuration system used for experiment management.

Configuration Hierarchy
-----------------------

The configuration system uses a hierarchical structure where settings can be
defined at multiple levels and overridden as needed:

.. code-block:: text

   conf/
   ├── config.yaml          # Base configuration (defaults)
   ├── solver/
   │   ├── fv.yaml          # Finite Volume solver settings
   │   └── spectral.yaml    # Spectral solver settings
   ├── experiment/
   │   ├── quick_test.yaml      # Fast debugging
   │   ├── sweep_test.yaml      # Sweep testing
   │   ├── fv_validation.yaml   # FV benchmark
   │   └── spectral_validation.yaml
   ├── mlflow/
   │   ├── local.yaml       # Local file tracking
   │   └── coolify.yaml     # Remote server
   └── hydra/
       └── launcher/
           └── joblib.yaml  # Parallel execution

Base Configuration
------------------

The main ``config.yaml`` defines default values for all parameters:

.. code-block:: yaml

   # conf/config.yaml
   defaults:
     - solver: fv
     - mlflow: local
     - _self_

   # Grid and physics
   N: 32                    # Grid size (cells for FV, polynomial order for spectral)
   Re: 100                  # Reynolds number
   lid_velocity: 1.0        # Lid velocity
   Lx: 1.0                  # Domain width
   Ly: 1.0                  # Domain height

   # Solver control
   tolerance: 1.0e-6        # Convergence tolerance
   max_iterations: 500      # Maximum iterations

   # Experiment tracking
   experiment_name: LDC-Solver
   sweep_name: sweep        # Parent run name for multirun sweeps

Solver Configurations
---------------------

Each solver has its own configuration file with solver-specific parameters.

**Finite Volume** (``conf/solver/fv.yaml``):

.. code-block:: yaml

   name: fv
   convection_scheme: upwind    # upwind, central, quick
   limiter: none                # none, minmod, vanLeer
   alpha_uv: 0.7                # Velocity under-relaxation
   alpha_p: 0.3                 # Pressure under-relaxation
   linear_solver_tol: 1.0e-6    # PETSc solver tolerance

**Spectral** (``conf/solver/spectral.yaml``):

.. code-block:: yaml

   name: spectral
   basis_type: chebyshev-gauss-lobatto  # Basis functions
   CFL: 0.5                             # CFL number for time stepping
   beta_squared: 1.0                    # Artificial compressibility
   corner_smoothing: true               # Smooth corner singularities

Experiment Configurations
-------------------------

Experiment configs override base settings for specific use cases. They use
``# @package _global_`` to merge into the root config.

**Quick Test** (``conf/experiment/quick_test.yaml``):

.. code-block:: yaml

   # @package _global_
   experiment_name: Quick-Test
   sweep_name: quick-test-sweep

   N: 16
   Re: 100
   tolerance: 1.0e-4
   max_iterations: 100

**Validation Sweep** (``conf/experiment/fv_validation.yaml``):

.. code-block:: yaml

   # @package _global_
   defaults:
     - override /solver: fv

   experiment_name: FV-Validation
   sweep_name: fv-validation-sweep

   N: 64
   Re: 100
   tolerance: 1.0e-7
   max_iterations: 50000

   # Define sweep parameters for multirun
   hydra:
     sweeper:
       params:
         N: 32,64,128
         Re: 100,400,1000

Creating Custom Experiments
---------------------------

To create a new experiment configuration:

1. Create a new YAML file in ``conf/experiment/``:

.. code-block:: yaml

   # conf/experiment/my_experiment.yaml
   # @package _global_

   experiment_name: My-Experiment
   sweep_name: my-sweep

   # Override any parameters
   N: 48
   Re: 400
   tolerance: 1.0e-8
   max_iterations: 10000

   # Optionally define sweep parameters
   hydra:
     sweeper:
       params:
         N: 32,48,64
         Re: 100,400

2. Run with your experiment:

.. code-block:: bash

   # Single run
   uv run python run_solver.py +experiment=my_experiment solver=fv

   # Sweep (uses hydra.sweeper.params if defined)
   uv run python run_solver.py -m +experiment=my_experiment

MLflow Integration
------------------

Experiment Name
^^^^^^^^^^^^^^^

The ``experiment_name`` field determines the MLflow experiment where runs are logged:

.. code-block:: yaml

   experiment_name: FV-Validation  # Creates/uses this MLflow experiment

Sweep Name (Parent Runs)
^^^^^^^^^^^^^^^^^^^^^^^^

When running parameter sweeps (``-m`` flag), a parent run is automatically created
to group all child runs. The ``sweep_name`` field controls the parent run's name:

.. code-block:: yaml

   sweep_name: fv-validation-sweep

This creates a hierarchy in MLflow:

.. code-block:: text

   fv-validation-sweep (parent)
   ├── fv_N32_Re100 (child)
   ├── fv_N32_Re400 (child)
   ├── fv_N64_Re100 (child)
   └── ...

You can also override the sweep name from the command line:

.. code-block:: bash

   uv run python run_solver.py -m sweep_name=custom-sweep solver=fv N=16,32,64

Command Line Usage
------------------

Basic Overrides
^^^^^^^^^^^^^^^

Override any parameter from the command line:

.. code-block:: bash

   # Override single parameters
   uv run python run_solver.py solver=spectral N=31 Re=1000

   # Override multiple parameters
   uv run python run_solver.py solver=fv N=64 Re=400 tolerance=1e-8

Using Experiments
^^^^^^^^^^^^^^^^^

Load an experiment configuration with ``+experiment=``:

.. code-block:: bash

   # Load experiment config
   uv run python run_solver.py +experiment=fv_validation

   # Load experiment and override parameters
   uv run python run_solver.py +experiment=fv_validation N=128 Re=1000

Parameter Sweeps
^^^^^^^^^^^^^^^^

Use ``-m`` (multirun) to sweep over parameters:

.. code-block:: bash

   # Sweep from command line
   uv run python run_solver.py -m solver=fv N=16,32,64 Re=100,400

   # Use experiment's predefined sweep
   uv run python run_solver.py -m +experiment=fv_validation

   # Parallel sweep with joblib
   uv run python run_solver.py -m hydra/launcher=joblib solver=fv,spectral N=16,32,64

Viewing Configuration
^^^^^^^^^^^^^^^^^^^^^

Print the resolved configuration without running:

.. code-block:: bash

   # Show resolved config
   uv run python run_solver.py --cfg job

   # Show config with experiment
   uv run python run_solver.py +experiment=fv_validation --cfg job
