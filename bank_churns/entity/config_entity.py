"""
Configuration entity classes for the Bank Churn prediction pipeline.
Uses dataclasses for clean, type-safe configuration management.
"""

import os
from dataclasses import dataclass
from datetime import datetime   
from bank_churns.constants import training_pipeline

@dataclass
class TrainingPipelineConfig:
    """
    Configuration for the overall training pipeline.
    Creates timestamped artifact directory for pipeline runs.
    """
    pipeline_name:str=training_pipeline.PIPLINE_NAME
    artifact_dir:str=training_pipeline.ARTIFACT_DIR
    timestamp:str=training_pipeline.TIMESTAMP

    def __post__int(self):
        """Create artifact directory structure after initialization."""
        self.artifact_path = os.path.join(self.artifact_dir, self.timestamp)
        os.makedirs(self.artifact_path, exist_ok=True)

    
@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion component.
    Handles MongoDB connection and train-test split settings.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init(self):
        self.data_base_name= training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.collection_name=training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # Directory setup
        self.data_ingestion_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
            )
        
        # File paths
        self.raw_data_file_path=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_RAW_DATA_FILE_NAME
        )
        self.train_file_path=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_TRAIN_FILE_NAME
        )
        self.test_file_path=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_TEST_FILE_NAME
        )

        # Split configuration
        self.train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.random_state = training_pipeline.DATA_INGESTION_RANDOM_STATE
        
        # Create directory
        os.makedirs(self.data_ingestion_dir, exist_ok=True)

@dataclass
class DataValidationConfig:
    """
    Configuration for data validation component.
    Validates data quality, schema compliance, and data drift.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        """Set up data validation paths."""
        self.data_validation_dir=os.path.join(
            self.training_pipeline_config.artifact_path,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.validation_report_file_path=os.path.join(
            self.training_pipeline_config.artifact_path,
            training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME
        )
        self.drift_report_file_path = os.path.join(
            self.training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DRFT_REPORT_FILE_NAME
        )

        # Expected schema
        self.expected_columns = training_pipeline.EXPECTED_COLUMNS
        self.columns_to_drop = training_pipeline.COLUMNS_TO_DROP
        self.target_column = training_pipeline.TARGET_COLUMN

        os.makedirs(self.data_validation_dir, exist_ok=True)

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation component.
    Handles feature engineering and preprocessing pipeline.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        """Set up data transformation paths and feature configurations."""
        self.data_transformation_dir = os.path.join(
            self.training_pipeline_config.artifact_path,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
        )

        self.transformed_train_file_path = os.path.join(
            self.transformed_data_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_NAME
        )

        self.transformed_test_file_path = os.path.join(
            self.transformed_data_dir,
            training_pipeline.DATA_TRANSFORMATION_TEST_FILE_NAME
        )

        self.preprocessor_obj_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_PREPROCESSOR_OBJ_FILE_NAME
        )

        # Feature configurations
        self.numerical_features = training_pipeline.NUMERICAL_FEATURES
        self.categorical_features = training_pipeline.CATEGORICAL_FEATURES
        self.binary_features = training_pipeline.BINARY_FEATURES
        self.target_column = training_pipeline.TARGET_COLUMN

        # Create directories
        os.makedirs(self.data_transformation_dir, exist_ok=True)
        os.makedirs(self.transformed_data_dir, exist_ok=True)

@dataclass
class ModelTrainerConfig:

   """
    Configuration for model training component.
    Handles model training parameters and quality thresholds.
    """
   training_pipeline_config: TrainingPipelineConfig

   def __post_init__(self):
       """Set up model training paths and parameters."""
       self.model_trainer_dir = os.path.join(
           self.training_pipeline_config.artifact_path,
           training_pipeline.MODEL_TRAINER_DIR_NAME
       )

       self.trained_model_file_path = os.path.join(
           self.model_trainer_dir,
           training_pipeline.MODEL_TRAINER_TRAINED_MODEL_FILE_NAME
       )

       # Model quality thresholds
       self.expected_score = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
       self.overfitting_underfitting_threshold = (
           training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
       )

       # Class imbalance handling
       self.use_class_weight = training_pipeline.USE_CLASS_WEIGHT

       os.makedirs(self.model_trainer_dir, exist_ok=True)


@dataclass
class ModelEvaluationConfig:
    """
    Configuration for model evaluation component.
    Compares new model against production model.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        """Set up model evaluation paths."""
        self.model_evaluation_dir = os.path.join(
            self.training_pipeline_config.artifact_path,
            training_pipeline.MODEL_EVALUATION_DIR_NAME
        )

        self.evaluation_report_file_path = os.path.join(
            self.model_evaluation_dir,
            training_pipeline.MODEL_EVALUATION_REPORT_FILE_NAME
        )

        self.changed_threshold_score = (
            training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        )

        os.makedirs(self.model_evaluation_dir, exist_ok=True)


@dataclass
class ModelPusherConfig:
    """
    Configuration for model pusher component.
    Deploys trained model to production directory.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        """Set up model pusher paths."""
        self.model_pusher_dir = os.path.join(
            self.training_pipeline_config.artifact_path,
            training_pipeline.MODEL_PUSHER_DIR_NAME
        )

        # Final production model directory
        self.saved_model_dir = training_pipeline.MODEL_PUSHER_SAVED_MODEL_DIR
        os.makedirs(self.saved_model_dir, exist_ok=True)

        self.final_model_file_path = os.path.join(
            self.saved_model_dir,
            training_pipeline.MODEL_FILE_NAME
        )

        self.final_preprocessor_file_path = os.path.join(
            self.saved_model_dir,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )

        os.makedirs(self.model_pusher_dir, exist_ok=True)
