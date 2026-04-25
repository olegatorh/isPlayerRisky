import os
import mlflow


def setup_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)