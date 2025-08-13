# airflow/dags/weather_data_collection_dag.py
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

from src.pipelines.predict_pipeline import prediction_pipeline

default_args = {
    'owner': 'data_scientist',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'prediction_pipeline_dag',
    default_args=default_args,
    description='Daily prediction pipeline for weather data',
    schedule_interval='0 2 * * *',  # Run at 2 AM every day
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=['weather', 'prediction'],
)

run_prediction = PythonOperator(
    task_id='run_prediction_pipeline',
    python_callable=prediction_pipeline,
    dag=dag,
)

run_prediction