Usage Guide
===========

This guide covers running solvers locally with Hydra configuration management
and on the DTU HPC cluster.

Hydra Configuration
-------------------

The project uses `Hydra <https://hydra.cc/>`_ for configuration management.
All solver runs are executed via ``run_solver.py``.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Finite Volume solver (32x32 cells, Re=100)
   uv run python run_solver.py solver=fv N=32 Re=100

   # Spectral solver (N=15 gives 16x16 nodes, Re=100)
   uv run python run_solver.py solver=spectral N=15 Re=100

Using Experiment Configs
^^^^^^^^^^^^^^^^^^^^^^^^

Pre-defined experiment configurations are in ``conf/experiment/``:

.. code-block:: bash

   # Quick test (small grid, few iterations)
   uv run python run_solver.py +experiment=quick_test solver=fv

   # FV validation (default settings for benchmarking)
   uv run python run_solver.py +experiment=fv_validation

   # Spectral validation
   uv run python run_solver.py +experiment=spectral_validation

Parameter Sweeps
^^^^^^^^^^^^^^^^

Run multiple configurations with Hydra's multirun (``-m``):

.. code-block:: bash

   # Sweep over grid sizes (sequential)
   uv run python run_solver.py -m solver=fv N=16,32,64 Re=100

   # Sweep over Reynolds numbers
   uv run python run_solver.py -m solver=spectral N=31 Re=100,400,1000

Parallel Sweeps (Joblib)
^^^^^^^^^^^^^^^^^^^^^^^^

Run sweeps in parallel using all CPU cores with the Joblib launcher:

.. code-block:: bash

   # Parallel sweep over grid sizes
   uv run python run_solver.py -m hydra/launcher=joblib solver=fv N=16,32,64 Re=100

   # Parallel sweep over solvers
   uv run python run_solver.py -m hydra/launcher=joblib solver=fv,spectral N=32 Re=100

   # Parallel multi-dimensional sweep (solver x N x Re = 12 jobs)
   uv run python run_solver.py -m hydra/launcher=joblib solver=fv,spectral N=16,32,64 Re=100,400

   # Control parallelism (e.g., 4 concurrent jobs)
   uv run python run_solver.py -m hydra/launcher=joblib hydra.launcher.n_jobs=4 solver=fv N=16,32,64

Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

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
   │   └── coolify.yaml         # Remote server
   └── hydra/
       └── launcher/
           └── joblib.yaml      # Parallel launcher (all cores)

Nested Runs for Sweeps
^^^^^^^^^^^^^^^^^^^^^^

Parameter sweeps automatically create a parent-child run hierarchy in MLflow:

- **Parent run**: Created before sweep starts, logs sweep configuration
- **Child runs**: Each parameter combination nested under the parent

This makes it easy to:

- View all runs from a sweep together in the MLflow UI
- Compare metrics across parameter combinations
- Track sweep-level metadata (HPC job ID, sweep config)

MLflow Tracking
^^^^^^^^^^^^^^^

Results are tracked with `MLflow <https://mlflow.org/>`_. Two modes available:

**Local Files (Default):**

.. code-block:: bash

   uv run python run_solver.py solver=fv mlflow=local

   # View UI
   uv run main.py --mlflow-ui

**Remote Server:**

.. code-block:: bash

   # Setup credentials (one-time)
   cp .env.template .env
   # Edit .env with your credentials

   # Run solver
   uv run python run_solver.py solver=fv mlflow=coolify

HPC Cluster (DTU)
-----------------

This section covers running parameter sweeps on the DTU HPC cluster using LSF.

Initial Setup
^^^^^^^^^^^^^

1. Clone the repository on the HPC cluster
2. Navigate into the repo root
3. Set up MLflow credentials:

.. code-block:: bash

   cp .env.template .env
   # Edit .env with your credentials

Submitting Jobs
^^^^^^^^^^^^^^^

Submit jobs using bsub with Hydra:

.. code-block:: bash

   # Single job
   bsub -q hpc -W 1:00 -n 4 -R "rusage[mem=4GB]" \
       "uv run python run_solver.py solver=fv N=32 Re=100 mlflow=coolify"

   # Sequential parameter sweep
   bsub -q hpc -W 4:00 -n 4 -R "rusage[mem=4GB]" \
       "uv run python run_solver.py -m solver=fv N=16,32,64 Re=100,400 mlflow=coolify"

Parallel Sweeps on HPC
^^^^^^^^^^^^^^^^^^^^^^

Use the Joblib launcher to run parameter combinations in parallel on a single node:

.. code-block:: bash

   # Parallel sweep using all cores on the node
   bsub -q hpc -W 2:00 -n 16 -R "rusage[mem=2GB]" -R "span[hosts=1]" \
       "uv run python run_solver.py -m hydra/launcher=joblib solver=fv,spectral N=16,32,64 Re=100,400 mlflow=coolify"

   # Control number of parallel jobs (e.g., 8 concurrent)
   bsub -q hpc -W 2:00 -n 8 -R "rusage[mem=4GB]" -R "span[hosts=1]" \
       "uv run python run_solver.py -m hydra/launcher=joblib hydra.launcher.n_jobs=8 solver=fv N=16,32,64,128 Re=100,400,1000 mlflow=coolify"

.. note::

   The ``-R "span[hosts=1]"`` flag ensures all cores are allocated on a single node.
   This is required because joblib uses local multiprocessing - it cannot distribute
   work across multiple nodes. Without this flag, LSF might split your cores across
   nodes, leaving some unusable.

Monitoring Jobs
^^^^^^^^^^^^^^^

Check the status of your running jobs:

.. code-block:: bash

   bstat

Example output:

.. code-block:: text

   JOBID      USER    QUEUE      JOB_NAME   NALLOC STAT  START_TIME      ELAPSED
   27198794   s214960 hpc        *N19-Re100      4 RUN   Nov 27 23:11    0:01:39
   27198795   s214960 hpc        *N23-Re100      4 RUN   Nov 27 23:11    0:01:39

Killing Jobs
^^^^^^^^^^^^

Kill jobs by name or ID:

.. code-block:: bash

   # Kill a specific job by name
   bkill -J LDC-N32-Re100

   # Kill a job by ID
   bkill 27198795

   # Kill all your jobs
   bkill 0

.. tip::

   Track job progress in the MLflow UI - each run logs the LSF job ID as a tag.
