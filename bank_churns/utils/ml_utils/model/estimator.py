"""
Model Estimator - Wrapper for preprocessing + model
"""
import sys
import numpy as np
import pandas as pd
from bank_churns.exception.exception import BankChurnException


class BankChurnModel:
    """
    Wrapper that combines feature engineering, preprocessing, and model.
    
    This is the COMPLETE inference pipeline:
    Raw Data → Feature Engineering → Preprocessing → Model → Prediction
    
    Why this architecture?
    - Single object for production inference
    - Ensures same transformations at train and inference time
    - No feature engineering code duplication
    """

    def __init__(self, preprocessor, model):
        """
        Initialize model wrapper.
        
        Args:
            preprocessor: Fitted ColumnTransformer
            model: Trained model
        """
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise BankChurnException(e, sys)

    def feature_engineering(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply same feature engineering as training.
        
        CRITICAL: This MUST match data_transformation.py exactly!
        """
        try:
            # 1. Log transformation for Age
            if 'Age' in dataframe.columns:
                dataframe['Age_log'] = np.log1p(dataframe['Age'])

            # 2. Balance to Salary ratio
            if 'Balance' in dataframe.columns and 'EstimatedSalary' in dataframe.columns:
                dataframe['Balance_Salary_Ratio'] = (
                    dataframe['Balance'] / (dataframe['EstimatedSalary'] + 1)
                )

            # 3. Products per Tenure
            if 'NumOfProducts' in dataframe.columns and 'Tenure' in dataframe.columns:
                dataframe['Products_Per_Tenure'] = (
                    dataframe['NumOfProducts'] / (dataframe['Tenure'] + 1)
                )

            # 4. Active Member with Credit Card
            if 'IsActiveMember' in dataframe.columns and 'HasCrCard' in dataframe.columns:
                dataframe['Active_CreditCard'] = (
                    dataframe['IsActiveMember'] * dataframe['HasCrCard']
                )

            return dataframe

        except Exception as e:
            raise BankChurnException(e, sys)

    def predict(self, X):
        """
        Predict on raw input data.
        
        Pipeline:
        1. Convert to DataFrame if needed
        2. Apply feature engineering
        3. Apply preprocessing
        4. Predict with model
        
        Args:
            X: Raw input features (can be array or DataFrame)
               Should have 11 columns BEFORE feature engineering
        
        Returns:
            Predictions array
        """
        try:
            # Step 1: Convert to DataFrame if numpy array
            if isinstance(X, np.ndarray):
                # Define column names (must match training data)
                columns = [
                    'CreditScore', 'Geography', 'Gender', 'Age', 
                    'Tenure', 'Balance', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
                ]
                X = pd.DataFrame(X, columns=columns)
            # Step 2: Apply feature engineering
            X_engineered = self.feature_engineering(X.copy())

            # Step 3: Apply preprocessing
            X_transformed = self.preprocessor.transform(X_engineered)

            # Step 4: Predict
            predictions = self.model.predict(X_transformed)

            return predictions

        except Exception as e:
            raise BankChurnException(e, sys)

    def predict_proba(self, X):
        """
        Predict probabilities on raw input data.
        
        Args:
            X: Raw input features
        
        Returns:
            Probability predictions
        """
        try:
            # Same pipeline as predict
            if isinstance(X, np.ndarray):
                columns = [
                    'CreditScore', 'Geography', 'Gender', 'Age', 
                    'Tenure', 'Balance', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
                ]
                X = pd.DataFrame(X, columns=columns)

            X_engineered = self.feature_engineering(X.copy())
            X_transformed = self.preprocessor.transform(X_engineered)
            probabilities = self.model.predict_proba(X_transformed)

            return probabilities

        except Exception as e:
            raise BankChurnException(e, sys)