Usage Guide
===========

This guide covers running solvers locally with Hydra configuration management
and on the DTU HPC cluster.

Quick Start
-----------

.. code-block:: bash

   # Run FV solver with validation experiment
   uv run python main.py -m +experiment/validation/ghia=fv

   # Run spectral solver
   uv run python main.py -m +experiment/validation/ghia=spectral

   # Single run (testing)
   uv run python main.py solver=fv N=32 Re=100

   # Regenerate plots without re-solving
   uv run python main.py -m +experiment/validation/ghia=fv plot_only=true

   # View MLflow UI
   uv run mlflow ui

Hydra Configuration
-------------------

The project uses `Hydra <https://hydra.cc/>`_ for configuration management.

Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   conf/
   ├── config.yaml              # Main config (N, Re, tolerance)
   ├── problem/
   │   └── ldc.yaml             # Physics (Re, domain size)
   ├── solver/
   │   ├── fv.yaml              # Finite Volume settings
   │   └── spectral/            # Spectral solver variants
   │       ├── sg.yaml          # Single Grid
   │       ├── fsg.yaml         # Full Single Grid MG
   │       ├── vmg.yaml         # V-cycle MultiGrid
   │       └── fmg.yaml         # Full MultiGrid
   ├── experiment/
   │   └── validation/ghia/     # Ghia benchmark experiments
   │       ├── fv.yaml
   │       └── spectral.yaml
   └── mlflow/
       ├── local.yaml           # File-based tracking (default)
       └── coolify.yaml         # Remote server

Parameter Sweeps
^^^^^^^^^^^^^^^^

Run multiple configurations with Hydra's multirun (``-m``):

.. code-block:: bash

   # Sweep over grid sizes
   uv run python main.py -m solver=fv N=16,32,64 Re=100

   # Sweep over Reynolds numbers
   uv run python main.py -m solver=fv N=32 Re=100,400,1000

   # Multi-dimensional sweep
   uv run python main.py -m solver=fv,spectral/sg N=16,32 Re=100,400

Sweeps automatically create parent-child run hierarchies in MLflow for easy comparison.

MLflow Tracking
^^^^^^^^^^^^^^^

Results are tracked with `MLflow <https://mlflow.org/>`_:

.. code-block:: bash

   # Local file-based tracking (default)
   uv run python main.py solver=fv N=32 Re=100

   # View results
   uv run mlflow ui
   # Open http://localhost:5000

   # Remote server (configure .env first)
   uv run python main.py solver=fv N=32 Re=100 mlflow=coolify

HPC Cluster (DTU LSF)
---------------------

Running experiments on the DTU HPC cluster using LSF job arrays.

Setup
^^^^^

1. Clone the repository on the HPC cluster
2. Configure MLflow credentials:

.. code-block:: bash

   cp .env.template .env
   # Edit .env with your credentials

Submitting Experiments
^^^^^^^^^^^^^^^^^^^^^^

Use the ``hpc_submit.py`` script to submit experiment sweeps as job arrays:

.. code-block:: bash

   # Preview job script (dry run)
   uv run python scripts/hpc_submit.py +experiment/validation/ghia=fv --dry-run

   # Submit FV validation (2 jobs: N=32, N=64)
   uv run python scripts/hpc_submit.py +experiment/validation/ghia=fv

   # Submit with custom resources
   uv run python scripts/hpc_submit.py +experiment/validation/ghia=spectral \
       --queue gpuv100 --time 4:00 --cores 8 --mem 8GB

The script:

1. Parses sweep parameters from the experiment config
2. Generates all parameter combinations
3. Submits an LSF job array where each job runs one configuration

Example output:

.. code-block:: text

   Parsing experiment: +experiment/validation/ghia=fv
   Sweep parameters: {'N': ['32', '64'], 'Re': ['100']}
   Total jobs: 2
     [1] {'N': '32', 'Re': '100'}
     [2] {'N': '64', 'Re': '100'}

Job Management
^^^^^^^^^^^^^^

.. code-block:: bash

   bstat          # Check job status
   bkill 12345    # Kill specific job
   bkill 0        # Kill all your jobs

Logs are written to ``logs/<job_id>_<array_index>.out``.
