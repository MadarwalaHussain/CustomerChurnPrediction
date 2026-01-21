"""
Model Evaluation Component with MLflow tracking
"""

import sys
import os

# STEP 1: Load environment FIRST
from dotenv import load_dotenv
load_dotenv()

# STEP 2: Apply SSL fix BEFORE mlflow
try:
    import fix_ssl
    fix_ssl.apply_ssl_fix()
    print(" SSL fix applied in model_evaluation.py")
except ImportError:
    print("  fix_ssl not found")

# STEP 3: Additional SSL settings
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# STEP 4: NOW import mlflow
import mlflow
import dagshub

# STEP 5: Import remaining libraries
import numpy as np

# Project imports
from bank_churns.entity.config_entity import ModelEvaluationConfig
from bank_churns.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact
)
from bank_churns.exception.exception import BankChurnException
from bank_churns.logging.logger import logging
from bank_churns.utils.main_utils.utils import load_object, load_numpy_array, write_yaml_file
from bank_churns.utils.ml_utils.metric.classification_metric import get_classification_score

# Get MLflow credentials
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')

# Initialize DagsHub
if mlflow_uri:
    try:
        dagshub.init(
            repo_owner='MadarwalaHussain',
            repo_name='CustomerChurnPrediction',
            mlflow=True
        )
        print(" DagsHub initialized in model_evaluation")
    except Exception as e:
        print(f"  DagsHub init failed: {e}")


class ModelEvaluation:
    """
    Model Evaluation Component - Compare models and track with MLflow.
    
    What we log to MLflow:
    - New model metrics
    - Production model metrics (if exists)
    - Comparison results
    - Accept/reject decision
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        """Initialize Model Evaluation component."""
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("Model Evaluation component initialized")
        except Exception as e:
            raise BankChurnException(e, sys)

    def evaluate_model(self) -> ModelEvaluationArtifact:
        """Evaluate trained model and compare with production model."""
        try:
            logging.info("=" * 70)
            logging.info("STARTING MODEL EVALUATION")
            logging.info("=" * 70)

            # Setup MLflow
            mlflow_enabled = False
            if mlflow_uri:
                try:
                    mlflow.set_tracking_uri(mlflow_uri)
                    mlflow.set_experiment("bank-churn-evaluation")
                    mlflow_enabled = True
                    logging.info(f"✓ MLflow tracking enabled")
                except Exception as e:
                    logging.warning(f"MLflow setup failed: {e}")
                    mlflow_enabled = False

            # Start MLflow run
            if mlflow_enabled:
                mlflow.start_run(run_name="model_evaluation")

            try:
                # Load test data
                logging.info("\nStep 1: Loading test data...")
                test_arr = load_numpy_array(
                    self.data_transformation_artifact.transformed_test_file_path
                )
                X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
                logging.info(f"Test data loaded: {X_test.shape}")

                # Load trained model
                logging.info("\nStep 2: Loading trained model...")
                trained_model = load_object(
                    self.model_trainer_artifact.trained_model_file_path
                )
                model_name = self.model_trainer_artifact.model_name
                logging.info(f"Trained model loaded: {model_name}")

                # Evaluate trained model
                logging.info("\nStep 3: Evaluating trained model...")
                y_pred = trained_model.predict(X_test)
                trained_model_score = get_classification_score(y_test, y_pred)

                logging.info("Trained Model Performance:")
                logging.info(f"  F1 Score:  {trained_model_score.f1_score:.4f}")
                logging.info(f"  Precision: {trained_model_score.precision_score:.4f}")
                logging.info(f"  Recall:    {trained_model_score.recall_score:.4f}")
                logging.info(f"  Accuracy:  {trained_model_score.accuracy_score:.4f}")
                logging.info(f"  ROC-AUC:   {trained_model_score.roc_auc_score:.4f}")

                # Log new model metrics to MLflow
                if mlflow_enabled:
                    try:
                        mlflow.log_param("new_model_name", model_name)
                        mlflow.log_metric("new_f1", trained_model_score.f1_score)
                        mlflow.log_metric("new_precision", trained_model_score.precision_score)
                        mlflow.log_metric("new_recall", trained_model_score.recall_score)
                        mlflow.log_metric("new_accuracy", trained_model_score.accuracy_score)
                        mlflow.log_metric("new_roc_auc", trained_model_score.roc_auc_score)
                    except Exception as e:
                        logging.warning(f"Failed to log new model metrics: {e}")

                # Check for production model
                logging.info("\nStep 4: Checking for production model...")
                production_model_path = os.path.join("final_models", "model.pkl")

                is_model_accepted = False
                improved_score = 0.0
                current_model_score = 0.0

                if os.path.exists(production_model_path):
                    logging.info("Production model found. Loading for comparison...")

                    # Load production model
                    production_model = load_object(production_model_path)

                    # Evaluate production model
                    y_pred_prod = production_model.predict(X_test)
                    production_model_score_obj = get_classification_score(y_test, y_pred_prod)
                    current_model_score = production_model_score_obj.f1_score

                    logging.info("Production Model Performance:")
                    logging.info(f"  F1 Score:  {production_model_score_obj.f1_score:.4f}")
                    logging.info(f"  Precision: {production_model_score_obj.precision_score:.4f}")
                    logging.info(f"  Recall:    {production_model_score_obj.recall_score:.4f}")
                    logging.info(f"  Accuracy:  {production_model_score_obj.accuracy_score:.4f}")

                    # Log production model metrics to MLflow
                    if mlflow_enabled:
                        try:
                            mlflow.log_metric("prod_f1", production_model_score_obj.f1_score)
                            mlflow.log_metric("prod_precision", production_model_score_obj.precision_score)
                            mlflow.log_metric("prod_recall", production_model_score_obj.recall_score)
                            mlflow.log_metric("prod_accuracy", production_model_score_obj.accuracy_score)
                            mlflow.log_metric("prod_roc_auc", production_model_score_obj.roc_auc_score)
                        except Exception as e:
                            logging.warning(f"Failed to log production metrics: {e}")

                    # Compare models
                    improved_score = trained_model_score.f1_score - current_model_score
                    threshold = self.model_evaluation_config.changed_threshold_score

                    logging.info(f"\nComparison:")
                    logging.info(f"  Current F1:    {current_model_score:.4f}")
                    logging.info(f"  New F1:        {trained_model_score.f1_score:.4f}")
                    logging.info(f"  Improvement:   {improved_score:.4f}")
                    logging.info(f"  Threshold:     {threshold:.4f}")

                    # Decision logic
                    if improved_score > threshold:
                        is_model_accepted = True
                        logging.info(f" New model ACCEPTED (improvement {improved_score:.4f} > {threshold:.4f})")
                    else:
                        is_model_accepted = False
                        logging.info(f" New model REJECTED (improvement {improved_score:.4f} <= {threshold:.4f})")

                else:
                    logging.info("No production model found. First deployment.")
                    is_model_accepted = True
                    current_model_score = 0.0
                    improved_score = trained_model_score.f1_score
                    logging.info(" New model ACCEPTED (first deployment)")

                # Log comparison to MLflow
                if mlflow_enabled:
                    try:
                        mlflow.log_metric("improvement", improved_score)
                        mlflow.log_param("threshold", threshold)
                        mlflow.log_param("is_accepted", is_model_accepted)
                        mlflow.log_param("has_production_model", os.path.exists(production_model_path))
                        
                        # Log decision reason
                        if is_model_accepted:
                            reason = f"Accepted: improvement {improved_score:.4f} > threshold {threshold:.4f}"
                        else:
                            reason = f"Rejected: improvement {improved_score:.4f} <= threshold {threshold:.4f}"
                        mlflow.log_param("decision_reason", reason)
                        
                        logging.info(" Comparison logged to MLflow")
                    except Exception as e:
                        logging.warning(f"Failed to log comparison: {e}")

                # Generate evaluation report
                logging.info("\nStep 5: Generating evaluation report...")
                evaluation_report = {
                    'trained_model': {
                        'model_name': model_name,
                        'f1_score': float(trained_model_score.f1_score),
                        'precision': float(trained_model_score.precision_score),
                        'recall': float(trained_model_score.recall_score),
                        'accuracy': float(trained_model_score.accuracy_score),
                        'roc_auc': float(trained_model_score.roc_auc_score)
                    },
                    'production_model': {
                        'f1_score': float(current_model_score) if current_model_score > 0 else None,
                        'exists': os.path.exists(production_model_path)
                    },
                    'comparison': {
                        'improvement': float(improved_score),
                        'threshold': float(threshold),
                        'is_accepted': bool(is_model_accepted)
                    }
                }

                # Save report
                write_yaml_file(
                    file_path=self.model_evaluation_config.evaluation_report_file_path,
                    content=evaluation_report
                )
                logging.info(f"Report saved: {self.model_evaluation_config.evaluation_report_file_path}")

                # Create artifact
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_score=improved_score,
                    current_model_score=current_model_score,
                    new_model_score=trained_model_score.f1_score,
                    evaluation_report_file_path=self.model_evaluation_config.evaluation_report_file_path
                )

                logging.info("=" * 70)
                logging.info("MODEL EVALUATION COMPLETED")
                logging.info("=" * 70)
                logging.info(f"Decision: {'ACCEPT' if is_model_accepted else 'REJECT'}")
                logging.info(f"New Model F1: {trained_model_score.f1_score:.4f}")
                logging.info(f"Current Model F1: {current_model_score:.4f}")
                logging.info(f"Improvement: {improved_score:.4f}")
                logging.info("=" * 70)

                return model_evaluation_artifact

            finally:
                # End MLflow run
                if mlflow_enabled:
                    try:
                        mlflow.end_run()
                        logging.info("MLflow run ended")
                    except Exception:
                        pass

        except Exception as e:
            logging.error("=" * 70)
            logging.error("MODEL EVALUATION FAILED")
            logging.error("=" * 70)
            raise BankChurnException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Main method to orchestrate model evaluation."""
        return self.evaluate_model()
