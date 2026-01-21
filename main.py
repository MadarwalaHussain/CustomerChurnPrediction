import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from bank_churns.utils.main_utils.utils import save_object, load_object
from bank_churns.utils.ml_utils.model.estimator import BankChurnModel
from bank_churns.utils.ml_utils.metric.classification_metric import get_classification_score, log_detailed_classification_report, calculate_business_metrics

from bank_churns.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from bank_churns.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from bank_churns.components.data_ingestion import DataIngestion
from bank_churns.components.data_validation import DataValidation
from bank_churns.components.data_transformation import DataTransformation
from bank_churns.components.model_trainer import ModelTrainer
from bank_churns.logging.logger import logging
from bank_churns.exception.exception import BankChurnException
import sys

def main():
    try:
        logging.info("\n" + "=" * 70)
        logging.info("BANK CHURN PREDICTION - TRAINING PIPELINE")
        logging.info("=" * 70 + "\n")

        # Initialize training pipeline configuration
        logging.info("Initializing training pipeline configuration...")
        training_pipeline_config= TrainingPipelineConfig()
        logging.info(f'Pipeline Name: {training_pipeline_config.pipeline_name}')
        logging.info(f'Artifact Directory: {training_pipeline_config.artifact_path}')

        # ==================== DATA INGESTION ====================
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 1: DATA INGESTION")
        logging.info("=" * 70)

        data_ingestion_config=DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config
        )

        data_ingestion_artifact=data_ingestion.init_data_ingestion()
        logging.info("Data Ingestion completed successfully")
        logging.info(f"   Train file: {data_ingestion_artifact.trained_file_path}")
        logging.info(f"   Test file: {data_ingestion_artifact.test_file_path}")

        # ==================== DATA VALIDATION ====================
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 2: DATA VALIDATION")
        logging.info("=" * 70)
        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation= DataValidation(data_ingestion_artifact=data_ingestion_artifact, 
                                        data_validation_config=data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()

        if data_validation_artifact.validation_status:
            logging.info('Data Validation passed')

        else:
            logging.warning(" Data Validation completed with warnings")
            logging.warning(f"   Message: {data_validation_artifact.message}")

        # ==================== DATA TRANSFORMATION ====================
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 3: DATA TRANSFORMATION")
        logging.info("=" * 70)
        data_transformation_config=DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )

        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )

        data_transformation_artifact= data_transformation.initiate_data_transformation()
        logging.info(" Data Transformation completed successfully")
        logging.info(f"   Transformed train: {data_transformation_artifact.transformed_train_file_path}")
        logging.info(f"   Transformed test: {data_transformation_artifact.transformed_test_file_path}")
        logging.info(f"   Preprocessor: {data_transformation_artifact.preprocessor_object_file_path}")

        # ==================== MODEL TRAINING ====================
        logging.info("\n" + "=" * 70)
        logging.info("PHASE 4: MODEL TRAINING")
        logging.info("=" * 70)

        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact  = model_trainer.initiate_model()

        logging.info(" Model Training completed successfully")
        logging.info(f"   Model: {model_trainer_artifact.model_name}")
        logging.info(f"   Test F1: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")
        logging.info(f"   Model saved to: {model_trainer_artifact.trained_model_file_path}")
        
    except Exception as e:
        logging.error("\n" + "=" * 70)
        logging.error("PIPELINE EXECUTION FAILED")
        logging.error("=" * 70)
        raise BankChurnException(e, sys)


if __name__=='__main__':
    main()