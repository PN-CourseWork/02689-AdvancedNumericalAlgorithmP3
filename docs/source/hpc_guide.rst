HPC Guide
=========

This guide covers running parameter sweeps on the DTU HPC cluster using LSF job packs.

Initial Setup (Once)
--------------------

1. Clone the repository
2. Navigate into the repo root
3. Run ``uv run setup_mlflow.py``
4. When asked for host, paste in::

      https://dbc-6756e917-e5fc.cloud.databricks.com
5. When asked for password token, create one in Databricks:

   :menuselection:`Account --> Developer --> Create Token`




Submitting Jobs
---------------

Use the ``main.py`` CLI to submit jobs:

.. code-block:: bash

   # Submit spectral solver jobs
   uv run python main.py --hpc spectral

   # Submit FV solver jobs
   uv run python main.py --hpc fv

   # Submit all jobs
   uv run python main.py --hpc all

Configuring Sweeps
------------------

Edit ``Experiments/*/generate_pack.sh`` to customize the parameter sweep:

.. code-block:: bash

   # Resource settings
   QUEUE="hpc"
   WALLTIME="1:00"
   CORES=4
   MEMORY="8GB"

   # Parameter sweep values
   N_VALUES=(11 15 19 23 27 31)
   RE_VALUES=(100 400 1000)

.. note::

   You may need to request longer wall time. The current example requests one hour, after which the job will terminate.

Monitoring Jobs
---------------

Check the status of your running jobs:

.. code-block:: bash

   bstat

Example output:

.. code-block:: text

   JOBID      USER    QUEUE      JOB_NAME   NALLOC STAT  START_TIME      ELAPSED
   27198794   s214960 hpc        *N19-Re100      4 RUN   Nov 27 23:11    0:01:39
   27198795   s214960 hpc        *N23-Re100      4 RUN   Nov 27 23:11    0:01:39
   27198792   s214960 hpc        *N11-Re100      4 RUN   Nov 27 23:11    0:01:39
   27198793   s214960 hpc        *N15-Re100      4 RUN   Nov 27 23:11    0:01:39

Killing Jobs
------------

Kill jobs by name or ID:

.. code-block:: bash

   # Kill a specific job by name
   bkill -J Spectral-N23-Re100

   # Kill a job by ID
   bkill 27198795

   # Kill all your jobs
   bkill 0

.. tip::

   See the job ID and name in the MLflow run description.
