from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def train_model():
    print("Training model...")

def deploy_model():
    print("Deploying model...")

with DAG(
    dag_id="mlops_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    task1 = PythonOperator(
        task_id="train",
        python_callable=train_model
    )

    task2 = PythonOperator(
        task_id="deploy",
        python_callable=deploy_model
    )

    task1 >> task2