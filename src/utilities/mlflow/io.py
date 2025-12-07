"""MLflow I/O utilities for experiment tracking."""

from pathlib import Path

import mlflow


def setup_mlflow_tracking(mode: str = "databricks"):
    """Configure MLflow tracking.

    Parameters
    ----------
    mode : str
        "databricks" or "local".
    """
    if mode == "databricks":
        try:
            mlflow.login(backend="databricks", interactive=False)
            mlflow.set_tracking_uri("databricks")
            print("INFO: Connected to Databricks MLflow tracking.")
        except Exception as e:
            raise RuntimeError(
                "MLflow Databricks setup failed. Ensure credentials are configured."
            ) from e
    elif mode == "local":
        mlruns_path = Path.cwd() / "mlruns"
        mlruns_uri = f"file://{mlruns_path}"
        mlflow.set_tracking_uri(mlruns_uri)
        print(f"INFO: Using local file-based MLflow tracking backend: {mlruns_uri}")
    else:
        print(
            f"WARNING: Unknown MLflow mode '{mode}'. Using existing URI: {mlflow.get_tracking_uri()}"
        )
