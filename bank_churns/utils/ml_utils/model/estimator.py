"""
Model estimator wrapper for the Bank Churn prediction system.
Combines preprocessor and model into a single prediction interface.
"""


import sys
import numpy as np
import pandas as pd
from typing import Optional
from bank_churns.exception.exception import BankChurnException
from bank_churns.logging.logger import logging


class BankChurnModel:
    """
    Wrapper class that combines the preprocessing and prediction
     Why this wrapper?
    - Clean abstraction: One interface for the entire pipeline
    - Production-ready: Easy to deploy (just load this object)
    - Type-safe: Handles DataFrames → arrays → predictions
    - Error handling: Catches issues at prediction time
    
    In production systems (AWS SageMaker, Azure ML), you typically need
    a single object that handles end-to-end inference. This is that object.
    
    Attributes:
        preprocessor: sklearn preprocessing pipeline (ColumnTransformer, Pipeline, etc.)
        model: Trained classifier (RandomForest, XGBoost, etc.)
    
    Example:
        >>> # During training
        >>> preprocessor = ColumnTransformer(...)
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train_transformed, y_train)
        >>> 
        >>> # Save as single object
        >>> bank_model = BankChurnModel(preprocessor, model)
        >>> save_object('final_models/model.pkl', bank_model)
        >>> 
        >>> # During inference
        >>> bank_model = load_object('final_models/model.pkl')
        >>> predictions = bank_model.predict(new_customer_data)
    """

    def __init__(self, preprocessor:object, model:object):
        """
        Initialize the model wrapper.
        
        Args:
            preprocessor: Fitted sklearn preprocessing pipeline
            model: Trained classification model
        
        Raises:
            BankChurnException: If preprocessor or model is None
        """
        try:
            if preprocessor is None:
                raise ValueError("Preprocessor cannot be None")
            if model is None:
                raise ValueError("Model cannot be None")

            self.preprocessor = preprocessor
            self.model = model
            logging.info("BankChurnModel initialized successfully")

        except Exception as e:
            raise BankChurnException(e, sys)

    def predict(self, X:pd.DataFrame)->np.ndarray:
        """
        Make predictions on new data.
        
        This method handles the complete inference pipeline:
        1. Validate input data
        2. Apply preprocessing (scaling, encoding, etc.)
        3. Make predictions with trained model
        4. Return predictions as array
        
        Args:
            X: Input DataFrame with same features as training data
               Should contain columns: CreditScore, Geography, Gender, Age, etc.
        
        Returns:
            NumPy array of predictions (0 or 1)
            0 = Customer will stay
            1 = Customer will churn
        
        Raises:
            BankChurnException: If prediction fails (missing columns, wrong dtype, etc.)
        
        Example:
            >>> new_customers = pd.DataFrame({
            ...     'CreditScore': [650, 720],
            ...     'Geography': ['France', 'Germany'],
            ...     'Gender': ['Male', 'Female'],
            ...     'Age': [35, 42],
            ...     'Tenure': [5, 3],
            ...     'Balance': [100000, 150000],
            ...     'NumOfProducts': [2, 1],
            ...     'HasCrCard': [1, 1],
            ...     'IsActiveMember': [1, 0],
            ...     'EstimatedSalary': [75000, 90000]
            ... })
            >>> predictions = bank_model.predict(new_customers)
            >>> print(predictions)  # [0, 1] - First stays, second churns
        """
        try:
            logging.info(f"starting prediction for {len(X)} samples")
        
            # validate input
            if X is None or len(X)==0:
                raise ValueError('input dataframe is empty')
            
            logging.info(f'Input shape: {X.shape}')
            logging.info(f'Input Columns: {X.columns.tolist()}')

            # step1: applying preprocessing
            logging.info('Applying preprocessing pipeline')
            X_transformed = self.preprocessor.transform(X)
            logging.info(f'Transformed shape {X_transformed.shape}')

            # Step2: make predictions
            logging.info("making the predictions with the trained model")
            predictions = self.model.predict(X_transformed)
            logging.info(f'predictions completed.Shape: {predictions.shape}')

            # log predictions distribution
            unique, counts = np.unique(predictions, return_counts=True)
            prediction_dist = dict(zip(unique,counts))
            logging.info(f'predictions distribution: {prediction_dist}')

            return predictions
        
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise BankChurnException(e, sys)


    def predict_proba(self, X:pd.DataFrame)->np.ndarray:
        """
        Predict class probabilities for new data.
        
        Useful when you need confidence scores instead of hard predictions.
        For example, you might only take action if churn probability > 0.7
        
        Args:
            X: Input DataFrame with same features as training data
        
        Returns:
            NumPy array of shape (n_samples, 2)
            - Column 0: Probability of class 0 (staying)
            - Column 1: Probability of class 1 (churning)
        
        Raises:
            BankChurnException: If prediction fails
            AttributeError: If model doesn't support predict_proba
        
        Example:
            >>> probabilities = bank_model.predict_proba(new_customers)
            >>> print(probabilities)
            # [[0.85, 0.15],  # 15% chance of churn
            #  [0.30, 0.70]]  # 70% chance of churn - take action!
            >>> 
            >>> # Get only churn probabilities
            >>> churn_probs = probabilities[:, 1]
            >>> high_risk = churn_probs > 0.7
        """

        try:
            logging.info(f'Starting probability predictions for {len(X)} samples')

            # Check if model supports predict_proba
            if not hasattr(self.model, 'predict_proba'):
                raise AttributeError(
                    f'Model {type(self.model).__name__} does not support predict_proba'
                )
            
            # Apply preprocessing
            logging.info("Applying preprocessing pipeline")
            X_transformed = self.preprocessor.transform(X)

            # Get probabilities
            logging.info("Calculating prediction probabilities")
            probabilities = self.model.predict_proba(X_transformed)
            logging.info(f"Probabilities calculated. Shape: {probabilities.shape}")

            # Log average churn probability
            avg_churn_prob = probabilities[:, 1].mean()
            logging.info(f"Average churn probability: {avg_churn_prob:.4f}")

            return probabilities

        except Exception as e:
            logging.error(f"Error during probability prediction: {str(e)}")
            raise BankChurnException(e, sys)
    
    
    def get_feature_names(self)->list:
        """
        Get feature names after preprocessing.
        
        Useful for feature importance analysis and debugging.
        
        Returns:
            List of feature names after transformation
        
        Example:
            >>> features = bank_model.get_feature_names()
            >>> print(features)
            # ['CreditScore', 'Age', 'Balance', 'Geography_France', 
            #  'Geography_Germany', 'Geography_Spain', 'Gender_Female', ...]
        """

        try:
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                return self.preprocessor.get_feature_names_out().tolist()
            else:
                logging.warning("Preprocessor does not have get_feature_names_out method")
                return[]
            
        except Exception as e:
            logging.error(f"Error getting feature names: {str(e)}")
            return []
    

    def __repr__(self) -> str:
        """
        Developer-friendly representation of the model.
        
        Returns:
            String representation with model and preprocessor types
        """
        return (
            f"BankChurnModel(\n"
            f"  preprocessor={type(self.preprocessor).__name__},\n"
            f"  model={type(self.model).__name__}\n"
            f")"
        )

    def __str__(self) -> str:
        """
        User-friendly string representation.
        
        Returns:
            Simple string describing the model
        """
        return f"Bank Churn Prediction Model (Preprocessor + {type(self.model).__name__})"



