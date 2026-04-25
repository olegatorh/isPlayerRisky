from datetime import datetime
from airflow.providers.standard.operators.bash import BashOperator
from airflow import DAG


with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2026, 4, 13),
    schedule="@daily",
    catchup=False,
    tags=["training", "is player risky"],
) as dag:
    training_pipeline_task = BashOperator(
        task_id="run_training_pipeline",
        bash_command="cd /opt/project && PYTHONPATH=/opt/project python -m src.pipeline.training_pipeline"
    )