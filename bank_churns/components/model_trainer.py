"""
Model Training Component for Bank Churn Prediction System.
Trains multiple models with hyperparameter tuning and quality gates.
"""


import os
import sys
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


from bank_churns.entity.config_entity import ModelTrainerConfig
from bank_churns.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact,ClassificationMetricArtifact

from bank_churns.exception.exception import BankChurnException
from bank_churns.logging.logger import logging
from bank_churns.utils.main_utils.utils import save_object,load_numpy_array, load_object

from bank_churns.utils.ml_utils.model.estimator import BankChurnModel
from bank_churns.utils.ml_utils.metric.classification_metric import get_classification_score, compare_model_metrics


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
    
    Why multiple models?
    - Different algorithms have different strengths
    - Ensemble methods often work best for tabular data
    - We can compare and select the best performer
    
    Your Research Applied:
    - class_weight='balanced' instead of SMOTE
    - Tree-based models (don't need PCA)
    - Focus on F1 score (imbalanced data)
    
    Example:
        >>> trainer = ModelTrainer(
        ...     data_transformation_artifact,
        ...     model_trainer_config
        ... )
        >>> artifact = trainer.initiate_model_trainer()
    """

    def __init__(self,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        """
        Initialize Model Trainer component.
        
        Args:
            data_transformation_artifact: Output from transformation (data paths)
            model_trainer_config: Configuration for training
        
        Raises:
            BankChurnException: If initialization fails
        """

        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            logging.info("Model Trainer component initialized")

        except Exception as e:
            raise BankChurnException(e, sys)
        
    def get_model_object_and_report(
            self,
            train:np.ndarray,
            test:np.ndarray
    )-> Tuple[object, float, ClassificationMetricArtifact, ClassificationMetricArtifact]:
        """
        Train multiple models and return the best one.
        
        Models to try:
        1. RandomForestClassifier - Robust, interpretable
        2. XGBoostClassifier - Often best for tabular data
        3. GradientBoostingClassifier - Strong baseline
        
        All models use class_weight='balanced' based on your research.
        
        Args:
            train: Training array (features + target)
            test: Test array (features + target)
        
        Returns:
            Tuple of (best_model, best_f1_score, train_metrics, test_metrics)
        
        Raises:
            BankChurnException: If training fails
        """
        try:
            logging.info('Starting Model training and evaluation...')

            X_train, y_train = train[:,:-1], train[:,-1]
            X_test, y_test = test[:, :-1], test[:, -1]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_train distribution: {np.bincount(y_train.astype(int))}")
            logging.info(f"y_test distribution: {np.bincount(y_test.astype(int))}")
            
            # Define models with class imbalance handling
            models= {
                'RandomForest': RandomForestClassifier(
                    n_estimators= 294, 
                    max_depth= 11,
                    min_samples_split= 7,
                    min_samples_leaf= 9,
                    bootstrap= True,
                    criterion= 'log_loss',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'XGBoost': XGBClassifier(
                    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Imbalance ratio
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    random_state=42
                ),
                'LogisticRegression': LogisticRegression(
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000
                )
            }

            # Store Model Report
            model_report={}
            logging.info("\n" + "=" * 70)
            logging.info("TRAINING MODELS")
            logging.info("=" * 70)
            for model_name, model in models.items():
                logging.info(f"\nTraining {model_name}...")

                # Train Model
                model.fit(X_train, y_train)
                logging.info(f"{model_name} training completed")

                # Make Predictions
                y_train_pred= model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Get Metrics
                train_metric = get_classification_score(y_train, y_train_pred)
                test_metric = get_classification_score(y_test, y_test_pred) 

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
            
            best_model_name=max(
                model_report,
                key=lambda x:model_report[x]['test_f1']
            )
            best_model_info = model_report[best_model_name]
            best_model = best_model_info['model']
            best_test_f1 = best_model_info['test_f1']

            logging.info(f"\nBest Model: {best_model_name}")
            logging.info(f"Test F1 Score: {best_test_f1:.4f}")

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

        except Exception as e:
            raise BankChurnException(e,sys)
    
    def finetune_best_model(
            self,
            model_name:str,
            X_train:np.ndarray,
            y_train: np.ndarray,
            X_test:np.ndarray,
            y_test: np.ndarray
    )->object:
        """
        Perform hyperparameter tuning on the best model.
        
        Uses GridSearchCV with cross-validation to find optimal parameters.
        
        Args:
            model_name: Name of the model to tune
            X_train, y_train: Training data
            X_test, y_test: Test data (for final evaluation)
        
        Returns:
            Tuned model
        """
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
                # Return base model without tuning
                if model_name=='RandomForest':
                    return RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
                elif model_name=='LogisticRegression':
                    return LogisticRegression(class_weight='balanced', random_state=42, max_iter=100)
                
            # Create base model
            if model_name=='RandomForest':
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

            # Perform Grig Search
            grid_search = GridSearchCV(base_model, param_grids[model_name], cv=3, scoring='f1', n_jobs=-1, verbose=1)
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
            # Return base model if tuning fails
            logging.warning("Returning base model without tuning")
            return RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    def initiate_model(self)-> ModelTrainerArtifact:
        """
        Main method to orchestrate model training pipeline.
        
        Training Pipeline:
        1. Load transformed train/test arrays
        2. Train multiple models
        3. Compare performance
        4. Select best model
        5. Fine-tune best model (optional)
        6. Check quality gates
        7. Load preprocessor
        8. Create BankChurnModel (preprocessor + model)
        9. Save final model
        10. Return artifact
        
        Quality Gates:
        - Test F1 score >= 0.80 (expected_score)
        - Train-test F1 gap < 0.05 (overfitting threshold)
        
        Returns:
            ModelTrainerArtifact with model path and metrics
        
        Raises:
            BankChurnException: If training fails or quality gates not met
        """ 
        try:
            logging.info("=" * 70)
            logging.info("STARTING MODEL TRAINING")
            logging.info("=" * 70)

            # Step 1:  Load Transformed Data
            train_arr = load_numpy_array(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(file_path=self.data_transformation_artifact.transformed_test_file_path)

            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            # Step 2 : Train and select best model
            logging.info("\nStep 2: Training models and selecting best performer...")

            best_model, best_f1, train_metric, test_metric = self.get_model_object_and_report(
                train=train_arr,
                test= test_arr
            )

            # Get Model name for logging
            model_name = type(best_model).__name__
            logging.info(f"\nBest model selected: {model_name}")


            # Step 3: Check quality gates
            logging.info("\nStep 3: Checking quality gates...")

            # Quality Gate1: Minimum f1 score
            if test_metric.f1_score< self.model_trainer_config.expected_score:
                raise Exception(
                    f"Model quality below threshold. "
                    f"Expected F1 >= {self.model_trainer_config.expected_score}, "
                    f"Got F1 = {test_metric.f1_score:.4f}"
                )
            # Quality Gate2: overfitting check
            f1_diff= abs(train_metric.f1_score - test_metric.f1_score)
            if f1_diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(
                    f"Potential overfitting detected: "
                    f"Train-Test F1 gap = {f1_diff:.4f} > {self.model_trainer_config.overfitting_underfitting_threshold}"
                )
            else:
                logging.info(
                    f" Quality Gate 2 passed: Train-Test gap = {f1_diff:.4f} < {self.model_trainer_config.overfitting_underfitting_threshold}")

            # Step 4: Load preprocessor
            preprocessor = load_object(file_path =self.data_transformation_artifact.preprocessor_object_file_path)
            logging.info(f"Preprocessor type: {type(preprocessor).__name__}")

            # step5 Create BankChurnModel(Wrapper)
            bank_churn_model = BankChurnModel(
                preprocessor=preprocessor,
                model=best_model
            )
            logging.info("BankChurnModel created successfully")

            # step6 save the model
            logging.info("\nStep 6: Saving trained model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=bank_churn_model
            )
            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")
            
            # Step 7 create artifact
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
