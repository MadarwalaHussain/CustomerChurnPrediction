"""
Load churn.csv into SQLite database
"""

import pandas as pd
import sqlite3
import os

DB_PATH = 'database/churn.db'


def load_training_data():
    """Load churn.csv into training_data table"""

    print("=" * 70)
    print("LOADING DATA INTO SQLITE")
    print("=" * 70)

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        print("Run: python database/init_sqlite.py first")
        return

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    print(f"✓ Connected to database: {DB_PATH}")

    # Load CSV
    csv_path = 'data/churn.csv'
    print(f"\nLoading CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records from CSV")

    # Rename columns to match database schema
    df_renamed = df.rename(columns={
        'CustomerId': 'customer_id',
        'CreditScore': 'credit_score',
        'Geography': 'geography',
        'Gender': 'gender',
        'Age': 'age',
        'Tenure': 'tenure',
        'Balance': 'balance',
        'NumOfProducts': 'num_of_products',
        'HasCrCard': 'has_credit_card',
        'IsActiveMember': 'is_active_member',
        'EstimatedSalary': 'estimated_salary',
        'Exited': 'exited'
    })

    # Drop unnecessary columns
    df_renamed = df_renamed.drop(columns=['RowNumber', 'Surname'], errors='ignore')

    # Check if data already exists
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM training_data")
    existing_count = cursor.fetchone()[0]

    if existing_count > 0:
        print(f"\n Warning: training_data already has {existing_count} records")
        response = input("Do you want to append data? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            conn.close()
            return

    # Load into database
    print(f"\nLoading into training_data table...")
    df_renamed.to_sql(
        'training_data',
        conn,
        if_exists='append',
        index=False,
        chunksize=1000
    )

    print(f"Loaded {len(df_renamed)} records")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM training_data")
    total = cursor.fetchone()[0]
    print(f"Total records in training_data: {total}")

    # Show statistics
    cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(exited) as churned,
        ROUND(AVG(exited) * 100, 2) as churn_rate_pct
    FROM training_data
    """)
    stats = cursor.fetchone()
    print(f"\nDatabase Statistics:")
    print(f"  Total customers: {stats[0]}")
    print(f"  Churned: {stats[1]}")
    print(f"  Churn rate: {stats[2]}%")

    # Show sample
    print("\nSample data:")
    sample = pd.read_sql("SELECT * FROM training_data LIMIT 5", conn)
    print(sample[['customer_id', 'geography', 'age', 'balance', 'exited']])

    conn.close()

    print("\n" + "=" * 70)
    print(" DATA LOAD COMPLETE")
    print("=" * 70)
    print(f"Database location: {os.path.abspath(DB_PATH)}")


if __name__ == '__main__':
    load_training_data()
