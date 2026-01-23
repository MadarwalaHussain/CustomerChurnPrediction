"""
Bank Churn Predictions DAG
Runs daily batch predictions
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'ml-team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='bank_churn_daily_predictions',
    default_args=default_args,
    description='Daily batch predictions for churn risk',
    schedule_interval='0 3 * * *',  # Every day at 3 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'prediction', 'batch']
) as dag:

    def fetch_new_customers():
        """Fetch new customers for prediction"""
        print("Fetching new customer data from database...")
        # TODO: Connect to your database
        # customers = fetch_from_db("SELECT * FROM customers WHERE predicted_date IS NULL")
        print("✓ Fetched 150 new customers")
        return 150

    def run_predictions(**context):
        """Run batch predictions"""
        num_customers = context['ti'].xcom_pull(task_ids='fetch_customers')
        print(f"Running predictions for {num_customers} customers...")

        # TODO: Load model and run predictions
        # model = load_production_model()
        # predictions = model.predict(customer_data)

        print("✓ Predictions completed")
        return num_customers

    def save_predictions(**context):
        """Save predictions to database"""
        num_customers = context['ti'].xcom_pull(task_ids='predict')
        print(f"Saving {num_customers} predictions to database...")

        # TODO: Save to database
        # save_to_db(predictions)

        print("✓ Predictions saved")

    def alert_high_risk():
        """Alert team about high-risk customers"""
        print("Checking for high-risk customers...")

        # TODO: Query predictions for high risk
        # high_risk = query_db("SELECT * FROM predictions WHERE churn_prob > 0.7")

        print("✓ Sent 12 alerts to retention team")

    fetch_task = PythonOperator(
        task_id='fetch_customers',
        python_callable=fetch_new_customers
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=run_predictions
    )

    save_task = PythonOperator(
        task_id='save_predictions',
        python_callable=save_predictions
    )

    alert_task = PythonOperator(
        task_id='alert_high_risk',
        python_callable=alert_high_risk
    )

    # Pipeline flow
    fetch_task >> predict_task >> save_task >> alert_task
