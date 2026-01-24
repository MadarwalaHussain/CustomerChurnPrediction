#!/bin/bash
# setup_database.sh

echo "========================================="
echo "Setting up SQLite Database"
echo "========================================="

# Create database directory
mkdir -p database

# Initialize database
echo "Initializing database..."
python database/init_sqlite.py

# Load data
echo "Loading data..."
python database/load_data_sqlite.py

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo "Database location: database/churn.db"
echo "You can now run your ML pipeline!"