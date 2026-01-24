"""
Initialize SQLite database with schema
"""

import sqlite3
import os

DB_PATH = 'database/churn.db'


def init_database():
    """Create database and tables"""

    print("=" * 70)
    print("INITIALIZING SQLITE DATABASE")
    print("=" * 70)

    # Create database directory
    os.makedirs('database', exist_ok=True)

    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f" Database file: {DB_PATH}")

    # ========================================
    # 1. TRAINING DATA TABLE
    # ========================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        credit_score INTEGER,
        geography TEXT,
        gender TEXT,
        age INTEGER,
        tenure INTEGER,
        balance REAL,
        num_of_products INTEGER,
        has_credit_card INTEGER,
        is_active_member INTEGER,
        estimated_salary REAL,
        exited INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    print(" Created table: training_data")

    # Indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_customer_id ON training_data(customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_exited ON training_data(exited)")

    # ========================================
    # 2. TEST DATA TABLE
    # ========================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL UNIQUE,
        credit_score INTEGER,
        geography TEXT,
        gender TEXT,
        age INTEGER,
        tenure INTEGER,
        balance REAL,
        num_of_products INTEGER,
        has_credit_card INTEGER,
        is_active_member INTEGER,
        estimated_salary REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_scored INTEGER DEFAULT 0,
        scored_at TIMESTAMP
    )
    """)
    print("Created table: test_data")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_customer_id ON test_data(customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_is_scored ON test_data(is_scored)")

    # ========================================
    # 3. PREDICTIONS TABLE
    # ========================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        churn_probability REAL NOT NULL,
        churn_prediction INTEGER NOT NULL,
        risk_category TEXT,
        model_version TEXT,
        prediction_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES test_data(customer_id)
    )
    """)
    print(" Created table: predictions")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON predictions(customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_category)")

    # ========================================
    # 4. MODEL METADATA TABLE
    # ========================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        algorithm TEXT,
        f1_score REAL,
        precision_score REAL,
        recall_score REAL,
        roc_auc_score REAL,
        training_date TIMESTAMP,
        training_records INTEGER,
        mlflow_run_id TEXT,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        promoted_at TIMESTAMP
    )
    """)
    print(" Created table: model_metadata")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_status ON model_metadata(status)")

    # ========================================
    # 5. MONITORING METRICS TABLE
    # ========================================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS monitoring_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT NOT NULL,
        metric_value REAL,
        metric_date DATE NOT NULL,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    print(" Created table: monitoring_metrics")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_date ON monitoring_metrics(metric_date)")

    # ========================================
    # 6. CREATE VIEWS
    # ========================================

    # High risk customers view
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS high_risk_customers AS
    SELECT 
        t.customer_id,
        t.geography,
        t.gender,
        t.age,
        t.balance,
        p.churn_probability,
        p.prediction_date
    FROM test_data t
    JOIN predictions p ON t.customer_id = p.customer_id
    WHERE p.risk_category = 'High'
    ORDER BY p.churn_probability DESC
    """)
    print(" Created view: high_risk_customers")

    # Monthly churn stats view
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS monthly_churn_stats AS
    SELECT 
        strftime('%Y-%m', prediction_date) AS month,
        COUNT(*) AS total_predictions,
        SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) AS predicted_churns,
        ROUND(AVG(churn_probability), 4) AS avg_churn_prob,
        COUNT(CASE WHEN risk_category = 'High' THEN 1 END) AS high_risk_count
    FROM predictions
    GROUP BY strftime('%Y-%m', prediction_date)
    ORDER BY month DESC
    """)
    print("✓ Created view: monthly_churn_stats")

    # Commit changes
    conn.commit()

    # Show database stats
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    print(f"\nDatabase initialized successfully!")
    print(f"  Location: {os.path.abspath(DB_PATH)}")
    print(f"  Size: {os.path.getsize(DB_PATH) / 1024:.2f} KB")
    print(f"  Tables: {len(tables)}")

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"    - {table[0]}: {count} records")

    conn.close()

    print("=" * 70)


if __name__ == '__main__':
    init_database()
