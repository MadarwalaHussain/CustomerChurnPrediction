"""
Model Training Component for Bank Churn Prediction System.
Trains multiple models with hyperparameter tuning and quality gates.
"""

# ====================================================================
# CRITICAL: IMPORT ORDER MATTERS!
# 1. Environment variables
# 2. SSL fix
# 3. HTTPS libraries (mlflow, dagshub)
# 4. Everything else
# ====================================================================

from bank_churns.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
    compare_model_metrics
)
from bank_churns.utils.ml_utils.model.estimator import BankChurnModel
from bank_churns.utils.main_utils.utils import save_object, load_numpy_array, load_object
from bank_churns.logging.logger import logging
from bank_churns.exception.exception import BankChurnException
from bank_churns.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ClassificationMetricArtifact
)
from bank_churns.entity.config_entity import ModelTrainerConfig
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Tuple
import numpy as np
from urllib.parse import urlparse
import dagshub
from mlflow.models import infer_signature
import mlflow
import urllib3
import ssl
import sys
import os

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


class ModelTrainer:
    """
    Model Training Component - Train and evaluate ML models.
    
    Responsibilities:
    1. Load transformed train/test data
    2. Train multiple models (RandomForest, XGBoost, etc.)
    3. Apply class imbalance handling (class_weight='balanced')
    4. Hyperparameter tuning
    5. Evaluate models with comprehensive metrics
    6. Check quality gates (F1 > 0.80, overfitting < 5%)
    7. Select best model
    8. Save model with preprocessor
    9. Log everything to MLflow
    
    Your Research Applied:
    - class_weight='balanced' instead of SMOTE
    - Tree-based models (don't need PCA)
    - Focus on F1 score (imbalanced data)
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        """Initialize Model Trainer component."""
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            logging.info("Model Trainer component initialized")
        except Exception as e:
            raise BankChurnException(e, sys)

    def get_model_object_and_report(
        self,
        train: np.ndarray,
        test: np.ndarray
    ) -> Tuple[object, float, ClassificationMetricArtifact, ClassificationMetricArtifact]:
        """Train multiple models and return the best one with MLflow tracking."""
        try:
            logging.info('Starting Model training and evaluation...')

            X_train, y_train = train[:, :-1], train[:, -1]
            X_test, y_test = test[:, :-1], test[:, -1]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_train distribution: {np.bincount(y_train.astype(int))}")
            logging.info(f"y_test distribution: {np.bincount(y_test.astype(int))}")

            # Setup MLflow (only if credentials available)
            mlflow_enabled = False
            if mlflow_uri:
                try:
                    mlflow.set_tracking_uri(mlflow_uri)
                    mlflow.set_experiment("bank-churn-training")
                    mlflow_enabled = True
                    logging.info(f" MLflow tracking enabled: {mlflow_uri}")
                except Exception as e:
                    logging.warning(f" MLflow setup failed: {e}")
                    logging.warning("  Continuing without MLflow tracking")
                    mlflow_enabled = False
            else:
                logging.info("MLflow not configured (MLFLOW_TRACKING_URI not set)")

            # Start MLflow run (if enabled)
            mlflow_run_started = False
            if mlflow_enabled:
                try:
                    mlflow.start_run(run_name="BestModel")
                    mlflow_run_started = True
                    logging.info("MLflow run started")
                except Exception as e:
                    logging.warning(f" Failed to start MLflow run: {e}")
                    mlflow_enabled = False

            try:
                # Define models with class imbalance handling
                models = {
                    'GradientBoosting': GradientBoostingClassifier(
                        n_estimators=200,        # Sara's optimized params
                        learning_rate=0.05,      # Sara's optimized params
                        max_depth=4,             # Sara's optimized params
                        subsample=0.8,           # Sara's optimized params
                        random_state=42
                    )
                }

                # Store Model Report
                model_report = {}

                logging.info("\n" + "=" * 70)
                logging.info("TRAINING MODELS")
                logging.info("=" * 70)

                for model_name, model in models.items():
                    logging.info(f"\nTraining {model_name}...")

                    # Train Model
                    model.fit(X_train, y_train)
                    logging.info(f"{model_name} training completed")

                    # Make Predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    # Get Metrics
                    train_metric = get_classification_score(y_train, y_train_pred)
                    test_metric = get_classification_score(y_test, y_test_pred)

                    # Log to MLflow (if enabled)
                    if mlflow_enabled:
                        try:
                            mlflow.log_param(f"{model_name}_model_type", model_name)
                            mlflow.log_metric(f"{model_name}_train_f1", float(train_metric.f1_score))
                            mlflow.log_metric(f"{model_name}_test_f1", float(test_metric.f1_score))
                            mlflow.log_metric(f"{model_name}_train_accuracy", float(train_metric.accuracy_score))
                            mlflow.log_metric(f"{model_name}_test_accuracy", float(test_metric.accuracy_score))
                            mlflow.log_metric(f"{model_name}_train_recall", float(train_metric.recall_score))
                            mlflow.log_metric(f"{model_name}_test_recall", float(test_metric.recall_score))
                            mlflow.log_metric(f"{model_name}_train_precision", float(train_metric.precision_score))
                            mlflow.log_metric(f"{model_name}_test_precision", float(test_metric.precision_score))
                            mlflow.log_metric(f"{model_name}_roc_auc", float(test_metric.roc_auc_score))

                            # Log confusion matrix
                            cm = confusion_matrix(y_test, y_test_pred)
                            mlflow.log_text(str(cm), f"{model_name}_confusion_matrix.txt")

                            logging.info(f"{model_name} metrics logged to MLflow")
                        except Exception as e:
                            logging.warning(f" Failed to log {model_name} to MLflow: {e}")

                    # Store report
                    model_report[model_name] = {
                        'model': model,
                        'train_f1': train_metric.f1_score,
                        'test_f1': test_metric.f1_score,
                        'train_metric': train_metric,
                        'test_metric': test_metric
                    }

                    logging.info(f"{model_name} Performance:")
                    logging.info(f"  Train F1: {train_metric.f1_score:.4f}")
                    logging.info(f"  Test F1:  {test_metric.f1_score:.4f}")
                    logging.info(f"  Difference: {abs(train_metric.f1_score - test_metric.f1_score):.4f}")

                # Select best model based on test F1 score
                logging.info("\n" + "=" * 70)
                logging.info("MODEL SELECTION")
                logging.info("=" * 70)

                best_model_name = max(
                    model_report,
                    key=lambda x: model_report[x]['test_f1']
                )
                best_model_info = model_report[best_model_name]
                best_model = best_model_info['model']
                best_test_f1 = best_model_info['test_f1']

                logging.info(f"\n Best Model: {best_model_name}")
                logging.info(f"  Test F1 Score: {best_test_f1:.4f}")

                # Log best model to MLflow
                if mlflow_enabled:
                    try:
                        mlflow.log_param("best_model", best_model_name)
                        mlflow.log_metric("best_test_f1", float(best_test_f1))

                        # Log model
                        signature = infer_signature(X_train, y_train)
                        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                        if tracking_url_type_store != 'file':
                            mlflow.sklearn.log_model(
                                best_model,
                                'model',
                                registered_model_name='BankChurnBestModelGB'
                            )
                            logging.info(" Model registered in MLflow Model Registry")
                        else:
                            mlflow.sklearn.log_model(
                                best_model,
                                'model',
                                signature=signature
                            )
                            logging.info(" Model logged to MLflow (local)")
                    except Exception as e:
                        logging.warning(f"  Failed to log best model to MLflow: {e}")

                # Log all model comparisons
                logging.info("\nModel Comparison:")
                for name, info in sorted(model_report.items(), key=lambda x: x[1]['test_f1'], reverse=True):
                    logging.info(f"  {name:20s} - Test F1: {info['test_f1']:.4f}")

                return (
                    best_model,
                    best_test_f1,
                    best_model_info['train_metric'],
                    best_model_info['test_metric']
                )

            finally:
                # End MLflow run (if started)
                if mlflow_run_started:
                    try:
                        mlflow.end_run()
                        logging.info(" MLflow run ended")
                    except Exception as e:
                        logging.warning(f"  Failed to end MLflow run: {e}")

        except Exception as e:
            logging.error(f" Error in model training: {str(e)}")
            raise BankChurnException(e, sys)

    def finetune_best_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> object:
        """Perform hyperparameter tuning on the best model."""
        try:
            logging.info(f"\nFine-tuning {model_name}...")

            # Define parameter grids for each model
            param_grids = {
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
            }

            if model_name not in param_grids:
                logging.info(f"No tuning parameters for {model_name}, returning base model")
                if model_name == 'RandomForest':
                    return RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
                elif model_name == 'LogisticRegression':
                    return LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

            # Create base model
            if model_name == 'RandomForest':
                base_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
            elif model_name == 'XGBoost':
                scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                base_model = XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            elif model_name == 'GradientBoosting':
                base_model = GradientBoostingClassifier(random_state=42)

            # Perform Grid Search
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            logging.info("Running GridSearchCV...")
            grid_search.fit(X_train, y_train)

            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

            # Evaluate on test set
            y_test_pred = grid_search.best_estimator_.predict(X_test)
            test_metric = get_classification_score(y_test, y_test_pred)
            logging.info(f"Test F1 score after tuning: {test_metric.f1_score:.4f}")

            return grid_search.best_estimator_

        except Exception as e:
            logging.error(f"Error in hyperparameter tuning: {str(e)}")
            logging.warning("Returning base model without tuning")
            return RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    def initiate_model(self) -> ModelTrainerArtifact:
        """Main method to orchestrate model training pipeline."""
        try:
            logging.info("=" * 70)
            logging.info("STARTING MODEL TRAINING")
            logging.info("=" * 70)

            # Step 1: Load Transformed Data
            logging.info("\nStep 1: Loading transformed data...")
            train_arr = load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            # Step 2: Train and select best model
            logging.info("\nStep 2: Training models and selecting best performer...")
            best_model, best_f1, train_metric, test_metric = self.get_model_object_and_report(
                train=train_arr,
                test=test_arr
            )

            # Get Model name for logging
            model_name = type(best_model).__name__
            logging.info(f"\n Best model selected: {model_name}")

            # Step 3: Check quality gates
            logging.info("\nStep 3: Checking quality gates...")

            # Quality Gate 1: Minimum F1 score
            if test_metric.f1_score < self.model_trainer_config.expected_score:
                raise Exception(
                    f"Model quality below threshold. "
                    f"Expected F1 >= {self.model_trainer_config.expected_score}, "
                    f"Got F1 = {test_metric.f1_score:.4f}"
                )
            logging.info(
                f"✓ Quality Gate 1 passed: F1 = {test_metric.f1_score:.4f} >= "
                f"{self.model_trainer_config.expected_score}"
            )

            # Quality Gate 2: Overfitting check
            f1_diff = abs(train_metric.f1_score - test_metric.f1_score)
            if f1_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(
                    f"  Potential overfitting detected: "
                    f"Train-Test F1 gap = {f1_diff:.4f} > "
                    f"{self.model_trainer_config.overfitting_underfitting_threshold}"
                )
            else:
                logging.info(
                    f" Quality Gate 2 passed: Train-Test gap = {f1_diff:.4f} < "
                    f"{self.model_trainer_config.overfitting_underfitting_threshold}"
                )

            # Step 4: Load preprocessor
            logging.info("\nStep 4: Loading preprocessor...")
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.preprocessor_object_file_path
            )
            logging.info(f"Preprocessor type: {type(preprocessor).__name__}")

            # Step 5: Create BankChurnModel (Wrapper)
            logging.info("\nStep 5: Creating BankChurnModel wrapper...")
            bank_churn_model = BankChurnModel(
                preprocessor=preprocessor,
                model=best_model
            )
            logging.info("BankChurnModel created successfully")

            # Step 6: Save the model
            logging.info("\nStep 6: Saving trained model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=bank_churn_model
            )
            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")

            # Step 7: Create artifact
            logging.info("\nStep 7: Creating model trainer artifact...")
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
                model_name=model_name
            )

            logging.info("=" * 70)
            logging.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
            logging.info("=" * 70)
            logging.info(f"Model: {model_name}")
            logging.info(f"Train F1: {train_metric.f1_score:.4f}")
            logging.info(f"Test F1:  {test_metric.f1_score:.4f}")
            logging.info(f"Train Accuracy: {train_metric.accuracy_score:.4f}")
            logging.info(f"Test Accuracy:  {test_metric.accuracy_score:.4f}")
            logging.info(f"Model saved to: {model_trainer_artifact.trained_model_file_path}")
            logging.info("=" * 70)

            return model_trainer_artifact

        except Exception as e:
            logging.error("=" * 70)
            logging.error("MODEL TRAINING FAILED")
            logging.error("=" * 70)
            raise BankChurnException(e, sys)
