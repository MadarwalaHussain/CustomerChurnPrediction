"""
Constants for the Bank Churn prediction training pipeline.
Centralized configuration for data ingestion, transformation, and model training.
"""

import os
from datetime import datetime

# Pipeline configuration
PIPELINE_NAME = 'bankchurns'
ARTIFACT_DIR = 'artifacts'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# DATA INGESTION
DATA_INGESTION_DIR_NAME = 'data_ingestion'
# DATA_INGESTION_COLLECTION_NAME ='bank_churns'
DATA_INGESTION_SOURCE_DATA_PATH = "churn.csv"
# DATA_INGESTION_DATABASE_NAME = 'bank_churns'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2
DATA_INGESTION_RANDOM_STATE  = 42

# File names
DATA_INGESTION_RAW_DATA_FILE_NAME='raw_data.csv'
DATA_INGESTION_TRAIN_FILE_NAME = 'train.csv'
DATA_INGESTION_TEST_FILE_NAME = 'test.csv'

# DATA VALIDATION
DATA_VALIDATION_DIR_NAME ='data_validation'
DATA_VALIDATION_REPORT_FILE_NAME='validation_report.yaml'
DATA_VALIDATION_DRFT_REPORT_FILE_NAME = 'drift_report.yaml'

# schema validation
SCHEMA_FILE_PATH = os.path.join('config', 'schema.yaml')

# Expected columns(excluding target and non-features)
EXPECTED_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary"
]

# Columns to drop (not used for modeling)
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

# Target column
TARGET_COLUMN = "Exited"

# Data Transformation 
DATA_TRANSFORMATION_DIR_NAME ='data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = 'transformed'
DATA_TRANSFORMATION_TRAIN_FILE_NAME = 'train.npy'
DATA_TRANSFORMATION_TEST_FILE_NAME = 'test.npy'
DATA_TRANSFORMATION_PREPROCESSOR_OBJ_FILE_NAME  = 'preprocessor.pkl'

# Feature engineering
NUMERICAL_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", 
    "NumOfProducts", "EstimatedSalary"
]

CATEGORICAL_FEATURES = ["Geography", "Gender"]

BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]

# Model Trainer
MODEL_TRAINER_DIR_NAME = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME = 'model.pkl'
MODEL_TRAINER_EXPECTED_SCORE  = 0.60 # minimum acceptable f1 score
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD = 0.05

# Model seletion
MODEL_CONFIG_FILE_PATH = os.path.join('config', 'model.yaml')

# Model evaluation
MODEL_EVALUATION_DIR_NAME = 'model_evaluation'
MODEL_EVALUATION_REPORT_FILE_NAME = 'evaluation_report.yaml'
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02 # Model improvement threshold

# Model Pusher 
MODEL_PUSHER_DIR_NAME = 'model_pusher'
MODEL_PUSHER_SAVED_MODEL_DIR = 'final_models'

# final model files
PREPROCESSING_OBJECT_FILE_NAME = 'preprocessing.pkl'
MODEL_FILE_NAME = 'model.pkl'

# MLFLOW
MLFLOW_EXPERIMENT_NAME = 'bank_churn_prediction'
MLFLOW_RUN_NAME = f'run_{TIMESTAMP}'

# Class imbalance handling
USE_CLASS_WEIGHT= True
SMOTE_SAMPLING_STRATEGY= None

# Feature Engg Constants
USE_LOG_TRANSFORMATION = True
AGE_LOG_FEATURE_NAME = 'Age_log'

# Feature interactions (examples)
FEATURE_INTERACTIONS = [
    ("Balance", "EstimatedSalary", "Balance_Salary_Ratio"),
    ("NumOfProducts", "IsActiveMember", "Products_Active_Interaction")
]

# PREDICTION CONSTANTS 
PREDICTION_OUTPUT_DIR = "prediction_output"
PREDICTION_OUTPUT_FILE_NAME = "predictions.csv"


