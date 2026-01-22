"""
Model Evaluation Component for Bank Churn Prediction System.
Compares trained model with production model and logs to MLflow.
"""
import dagshub
import pandas as pd
import mlflow
from bank_churns.utils.main_utils.utils import load_object, load_numpy_array, write_yaml_file
from bank_churns.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact
)
import os
import sys
import urllib3
import ssl
# STEP 1: Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# STEP 2: Apply SSL fix BEFORE any HTTPS imports
try:
    import fix_ssl
    fix_ssl.apply_ssl_fix()
    print("SSL fix applied in model_trainer.py")
except ImportError:
    print(" Warning: fix_ssl module not found")
except Exception as e:
    print(f" Warning: SSL fix failed: {e}")

# Additional SSL bypass (belt and suspenders approach)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variables for SSL bypass
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# STEP 3: NOW import HTTPS libraries (mlflow, dagshub)

# STEP 4: Import remaining libraries

# Project imports

# Get MLflow credentials from environment
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_user_name = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD')


# Initialize DagsHub (if credentials available)
if mlflow_uri and mlflow_user_name:
    try:
        dagshub.init(
            repo_owner='MadarwalaHussain',
            repo_name='CustomerChurnPrediction',
            mlflow=True
        )
        print("DagsHub initialized successfully")
    except Exception as e:
        print(f" DagsHub initialization failed: {e}")
        print("Continuing without DagsHub integration")



from bank_churns.entity.config_entity import ModelEvaluationConfig
from bank_churns.exception.exception import BankChurnException
from bank_churns.logging.logger import logging
from bank_churns.utils.ml_utils.metric.classification_metric import get_classification_score


class ModelEvaluation:
    """
    Model Evaluation Component - Compare models and track experiments.
    
    Responsibilities:
    1. Load trained model from Model Trainer
    2. Evaluate on test data
    3. Compare with production model (if exists)
    4. Log everything to MLflow/DagsHub
    5. Decide if new model should be deployed
    6. Generate evaluation report
    
    Why Model Evaluation?
    - Prevents deploying worse models
    - Tracks experiment history
    - Enables A/B testing
    - Provides audit trail
    
    MLflow Tracking:
    - All parameters logged
    - All metrics logged
    - Models versioned
    - Enables comparison
    
    Example:
        >>> evaluator = ModelEvaluation(
        ...     model_trainer_artifact,
        ...     model_evaluation_config,
        ...     data_transformation_artifact
        ... )
        >>> artifact = evaluator.initiate_model_evaluation()
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        """
        Initialize Model Evaluation component.
        
        Args:
            model_trainer_artifact: Output from model trainer
            model_evaluation_config: Configuration for evaluation
            data_transformation_artifact: For loading test data
        
        Raises:
            BankChurnException: If initialization fails
        """
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact

            # Initialize MLflow (if credentials available)
            self._setup_mlflow()

            logging.info("Model Evaluation component initialized")

        except Exception as e:
            raise BankChurnException(e, sys)

    def _setup_mlflow(self):
        """Setup MLflow tracking with DagsHub."""
        try:
            # Load environment variables
            load_dotenv()

            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

            if mlflow_uri:
                # Set MLflow tracking URI
                mlflow.set_tracking_uri(mlflow_uri)

                # Set experiment
                mlflow.set_experiment("bank-churn-prediction")

                logging.info(f"MLflow tracking enabled: {mlflow_uri}")
                self.mlflow_enabled = True
            else:
                logging.info("MLflow tracking not configured (MLFLOW_TRACKING_URI not set)")
                self.mlflow_enabled = False

        except Exception as e:
            logging.warning(f"MLflow setup failed: {e}. Continuing without tracking.")
            self.mlflow_enabled = False

    def evaluate_model(self) -> ModelEvaluationArtifact:
        """
        Evaluate trained model and compare with production model.
        
        Evaluation Logic:
        1. Load test data
        2. Load trained model
        3. Evaluate trained model on test set
        4. Check if production model exists
        5. If exists, compare performance
        6. Decide if new model should be accepted
        7. Log to MLflow
        8. Generate evaluation report
        
        Acceptance Criteria:
        - New model F1 > current model F1 + threshold (0.02)
        - Or no production model exists (first deployment)
        
        Returns:
            ModelEvaluationArtifact with acceptance decision
        
        Raises:
            BankChurnException: If evaluation fails
        """
        try:
            logging.info("=" * 70)
            logging.info("STARTING MODEL EVALUATION")
            logging.info("=" * 70)

            # Step 1: Load test data
            logging.info("\nStep 1: Loading test data...")
            # test_arr = load_numpy_array(
            #     self.data_transformation_artifact.transformed_test_file_path
            # )
            test_df = pd.read_csv(self.model_evaluation_config.new_test_file_path)
            X_test, y_test = test_df.drop(columns=['Exited']), test_df['Exited']
            logging.info(f"Test data loaded: {X_test.shape}")

            # Step 2: Load trained model
            logging.info("\nStep 2: Loading trained model...")
            trained_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )
            logging.info(f"Trained model loaded: {type(trained_model).__name__}")

            # Step 3: Evaluate trained model
            logging.info("\nStep 3: Evaluating trained model...")
            y_pred = trained_model.predict(X_test)
            trained_model_score = get_classification_score(y_test, y_pred)

            logging.info("Trained Model Performance:")
            logging.info(f"  F1 Score:  {trained_model_score.f1_score:.4f}")
            logging.info(f"  Precision: {trained_model_score.precision_score:.4f}")
            logging.info(f"  Recall:    {trained_model_score.recall_score:.4f}")
            logging.info(f"  Accuracy:  {trained_model_score.accuracy_score:.4f}")
            logging.info(f"  ROC-AUC:   {trained_model_score.roc_auc_score:.4f}")

            # Step 4: Check for production model
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
                logging.info("No production model found. This is the first deployment.")
                is_model_accepted = True
                current_model_score = 0.0
                improved_score = trained_model_score.f1_score
                logging.info(" New model ACCEPTED (first deployment)")

            # Step 5: Log to MLflow
            if self.mlflow_enabled:
                logging.info("\nStep 5: Logging to MLflow...")
                self._log_to_mlflow(
                    trained_model_score=trained_model_score,
                    current_model_score=current_model_score,
                    is_accepted=is_model_accepted,
                    improved_score=improved_score
                )

            # Step 6: Generate evaluation report
            logging.info("\nStep 6: Generating evaluation report...")
            evaluation_report = {
                'trained_model': {
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
                    'threshold': float(self.model_evaluation_config.changed_threshold_score),
                    'is_accepted': bool(is_model_accepted)
                }
            }

            # Save report
            write_yaml_file(
                file_path=self.model_evaluation_config.evaluation_report_file_path,
                content=evaluation_report
            )
            logging.info(f"Evaluation report saved: {self.model_evaluation_config.evaluation_report_file_path}")

            # Step 7: Create artifact
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
            logging.info(f"Decision: {'ACCEPT ' if is_model_accepted else 'REJECT '}")
            logging.info(f"New Model F1: {trained_model_score.f1_score:.4f}")
            logging.info(f"Current Model F1: {current_model_score:.4f}")
            logging.info(f"Improvement: {improved_score:.4f}")
            logging.info("=" * 70)

            return model_evaluation_artifact

        except Exception as e:
            logging.error("=" * 70)
            logging.error("MODEL EVALUATION FAILED")
            logging.error("=" * 70)
            raise BankChurnException(e, sys)

    def _log_to_mlflow(
        self,
        trained_model_score,
        current_model_score: float,
        is_accepted: bool,
        improved_score: float
    ):
        """Log evaluation metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=f"evaluation_{self.model_trainer_artifact.model_name}"):
                # Log model info
                mlflow.log_param("model_name", self.model_trainer_artifact.model_name)
                mlflow.log_param("evaluation_stage", "model_comparison")

                # Log new model metrics
                mlflow.log_metric("new_f1_score", trained_model_score.f1_score)
                mlflow.log_metric("new_precision", trained_model_score.precision_score)
                mlflow.log_metric("new_recall", trained_model_score.recall_score)
                mlflow.log_metric("new_accuracy", trained_model_score.accuracy_score)
                mlflow.log_metric("new_roc_auc", trained_model_score.roc_auc_score)

                # Log production model metric
                mlflow.log_metric("current_f1_score", current_model_score)

                # Log comparison
                mlflow.log_metric("improvement", improved_score)
                mlflow.log_param("is_accepted", is_accepted)
                mlflow.log_param("threshold", self.model_evaluation_config.changed_threshold_score)

                logging.info(" Metrics logged to MLflow")

        except Exception as e:
            logging.warning(f"Failed to log to MLflow: {e}")

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Main method to orchestrate model evaluation.
        
        Returns:
            ModelEvaluationArtifact with evaluation results
        """
        return self.evaluate_model()
