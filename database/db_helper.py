"""
Database helper class for easy database operations
"""

import sqlite3
import pandas as pd
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

DB_PATH = 'database/churn.db'


class DatabaseHelper:
    """Helper class for SQLite operations"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        """Context manager for database connection"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute a SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount

    def fetch_dataframe(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute query and return as pandas DataFrame"""
        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=params)

    def insert_dataframe(self, df: pd.DataFrame, table_name: str,
                         if_exists: str = 'append') -> int:
        """Insert pandas DataFrame into table"""
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            return len(df)

    # ========================================
    # Specific queries
    # ========================================

    def get_training_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get training data"""
        query = "SELECT * FROM training_data"
        if limit:
            query += f" LIMIT {limit}"
        return self.fetch_dataframe(query)

    def get_test_data(self, unscored_only: bool = True) -> pd.DataFrame:
        """Get test data (optionally only unscored)"""
        query = "SELECT * FROM test_data"
        if unscored_only:
            query += " WHERE is_scored = 0"
        return self.fetch_dataframe(query)

    def save_predictions(self, predictions_df: pd.DataFrame) -> int:
        """Save predictions to database"""
        return self.insert_dataframe(predictions_df, 'predictions')

    def mark_as_scored(self, customer_ids: List[int]) -> int:
        """Mark customers as scored"""
        placeholders = ','.join('?' * len(customer_ids))
        query = f"""
        UPDATE test_data 
        SET is_scored = 1, scored_at = CURRENT_TIMESTAMP
        WHERE customer_id IN ({placeholders})
        """
        return self.execute_update(query, tuple(customer_ids))

    def get_high_risk_customers(self) -> pd.DataFrame:
        """Get high risk customers from view"""
        return self.fetch_dataframe("SELECT * FROM high_risk_customers")

    def get_monthly_stats(self) -> pd.DataFrame:
        """Get monthly churn statistics"""
        return self.fetch_dataframe("SELECT * FROM monthly_churn_stats")

    def log_model_metadata(self, metadata: Dict[str, Any]) -> int:
        """Log model training metadata"""
        query = """
        INSERT INTO model_metadata 
        (model_name, model_version, algorithm, f1_score, precision_score, 
         recall_score, roc_auc_score, training_date, training_records, 
         mlflow_run_id, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            metadata.get('model_name'),
            metadata.get('model_version'),
            metadata.get('algorithm'),
            metadata.get('f1_score'),
            metadata.get('precision_score'),
            metadata.get('recall_score'),
            metadata.get('roc_auc_score'),
            metadata.get('training_date'),
            metadata.get('training_records'),
            metadata.get('mlflow_run_id'),
            metadata.get('status', 'training')
        )
        return self.execute_update(query, params)

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        stats = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Count records in each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

        return stats


# Example usage
# if __name__ == '__main__':
#     db = DatabaseHelper()

#     print("Database Statistics:")
#     stats = db.get_database_stats()
#     for table, count in stats.items():
#         print(f"  {table}: {count} records")

#     print("\nSample training data:")
#     sample = db.get_training_data(limit=5)
#     print(sample[['customer_id', 'geography', 'age', 'exited']])
