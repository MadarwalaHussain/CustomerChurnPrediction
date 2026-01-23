"""
Bank Churn Training DAG - CSV Version
Weekly model training using churn.csv
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
import os

# ========================================
# CONFIGURATION - UPDATE THIS PATH!
# ========================================
PROJECT_PATH = r'C:\Users\hmadarw\Learnings\DataScience\Projects\CustomerChurnPrediction'

# MLflow credentials
MLFLOW_TRACKING_URI = 'https://dagshub.com/MadarwalaHussain/CustomerChurnPrediction.mlflow'
MLFLOW_USERNAME = 'MadarwalaHussain'
MLFLOW_PASSWORD = 'db926fa2f90af5a877aa3d843baf4a9555d30c6c'
DAGSHUB_USERNAME = 'MadarwalaHussain'

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['hussainmadar4@gmail.com'],
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='bank_churn_training',
    default_args=default_args,
    description='Weekly bank customer churn model training',
    schedule_interval='0 2 * * 1',  # Every Monday 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'churn']
) as dag:

    # ========================================
    # TASK 1: Check Data
    # ========================================
    check_data = BashOperator(
        task_id='check_data_exists',
        bash_command=f"""
        echo "Project path: {PROJECT_PATH}"
        echo "Checking for data file..."
        if [ -f "{PROJECT_PATH}/data/churn.csv" ]; then
            echo "✓ Data file found"
            wc -l "{PROJECT_PATH}/data/churn.csv"
            exit 0
        else
            echo "❌ Data file not found"
            ls -la "{PROJECT_PATH}/data/"
            exit 1
        fi
        """
    )

    # ========================================
    # TASK 2: Validate Data
    # ========================================
    def validate_data():
        """Validate data quality"""
        import pandas as pd

        print("=" * 70)
        print("DATA VALIDATION")
        print("=" * 70)

        data_path = f"{PROJECT_PATH}/data/churn.csv"
        print(f"Loading: {data_path}")

        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} rows")

        # Validate
        min_rows = 1000
        if len(df) < min_rows:
            raise ValueError(f"Too few rows: {len(df)}")

        required_cols = ['CreditScore', 'Geography', 'Age', 'Exited']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        churn_rate = df['Exited'].mean()
        print(f"✓ Churn rate: {churn_rate:.2%}")

        print("=" * 70)
        print("✓ VALIDATION PASSED")
        print("=" * 70)

        return {'rows': len(df), 'churn_rate': churn_rate}

    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data
    )

    # ========================================
    # TASK 3: Train Model
    # ========================================
    train_model = DockerOperator(
        task_id='train_model',
        image='bank-churn-ml:latest',
        command='python main.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        auto_remove=True,
        environment={
            'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI,
            'MLFLOW_TRACKING_USERNAME': MLFLOW_USERNAME,
            'MLFLOW_TRACKING_PASSWORD': MLFLOW_PASSWORD,
            'DAGSHUB_USERNAME': DAGSHUB_USERNAME
        },
        mount_tmp_dir=False,
        mounts=[
            {
                'source': f"{PROJECT_PATH}\\data",
                'target': '/app/data',
                'type': 'bind'
            },
            {
                'source': f"{PROJECT_PATH}\\artifacts",
                'target': '/app/artifacts',
                'type': 'bind'
            },
            {
                'source': f"{PROJECT_PATH}\\logs",
                'target': '/app/logs',
                'type': 'bind'
            },
            {
                'source': f"{PROJECT_PATH}\\final_models",
                'target': '/app/final_models',
                'type': 'bind'
            }
        ]
    )

    # ========================================
    # TASK 4: Notify
    # ========================================
    def notify_completion(**context):
        """Send notification"""
        print("=" * 70)
        print("✓ TRAINING COMPLETED")
        print("=" * 70)

        stats = context['ti'].xcom_pull(task_ids='validate_data')
        if stats:
            print(f"Rows: {stats['rows']}")
            print(f"Churn: {stats['churn_rate']:.2%}")

        print("Check DagsHub for results")
        print("=" * 70)
        return "Success"

    notify = PythonOperator(
        task_id='notify',
        python_callable=notify_completion
    )

    # ========================================
    # Pipeline Flow
    # ========================================
    check_data >> validate_data_task >> train_model >> notify
