from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import timedelta
os.environ["DATA_ROOT"] = "/opt/data"
os.environ["MODEL_ROOT"] = "/opt/models"

# Add project root to Python path to import custom modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# Add project root to Python path to import custom modules
# sys.path.append('/opt')  # Ensure this matches the container's directory structure

from src.pipelines.drift_monitoring_pipeline import run_drift_monitoring_pipeline

default_args = {
    'owner': 'data_scientist',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'drift_detection',
    default_args=default_args,
    description='Drift detection for models',
    schedule_interval='0 1 * * *',  # Run at 1 AM every day
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=['drift', 'detection'],
)

run_data_collection = PythonOperator(
    task_id='run_data_collection',
    python_callable=run_drift_monitoring_pipeline,
    dag=dag,
)

run_data_collection