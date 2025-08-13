from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import timedelta
os.environ["DATA_ROOT"] = "/opt/data"
# Add project root to Python path to import custom modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# Add project root to Python path to import custom modules
# sys.path.append('/opt')  # Ensure this matches the container's directory structure

from src.pipelines.data_collection import run_data_collection_pipeline

default_args = {
    'owner': 'data_scientist',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'weather_data_collection',
    default_args=default_args,
    description='Daily weather data collection and preprocessing',
    schedule_interval='0 1 * * *',  # Run at 1 AM every day
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=['weather', 'data_collection'],
)

run_data_collection = PythonOperator(
    task_id='run_data_collection',
    python_callable=run_data_collection_pipeline,
    dag=dag,
)

run_data_collection