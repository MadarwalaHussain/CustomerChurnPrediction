"""
Data Transformation Component for Bank Churn Prediction System.
Handles feature engineering and preprocessing pipeline creation.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple

from bank_churns.entity.config_entity import DataTransformationConfig
from bank_churns.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from bank_churns.exception.exception import BankChurnException
from bank_churns.logging.logger import logging

from bank_churns.utils.main_utils.utils import save_object, save_numpy_array

class DataTransformation:
    """
    Data Transformation Component - Feature engineering and preprocessing.
    
    Responsibilities:
    1. Drop non-predictive columns (RowNumber, CustomerId, Surname)
    2. Feature engineering (log transformations, interactions)
    3. Create preprocessing pipeline (scaling, encoding)
    4. Transform train and test data
    5. Save transformed arrays and preprocessor
    
    Why this architecture?
    - Separation of feature engineering and preprocessing
    - Reusable preprocessor for production inference
    - Prevents data leakage (fit only on train)
    - Reproducible transformations
    
    Example:
        >>> transformation = DataTransformation(
        ...     data_validation_artifact,
        ...     data_transformation_config
        ... )
        >>> artifact = transformation.initiate_data_transformation()
        >>> print(artifact.transformed_train_file_path)
    
    """
    def __init__(self, data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        """
        Initialize Data Transformation component.
        
        Args:
            data_validation_artifact: Output from validation (contains data paths)
            data_transformation_config: Configuration for transformation
        
        Raises:
            BankChurnException: If initialization fails
        """
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            # We need to reconstruct data paths from validation artifact path
            # validation_report_file_path looks like: artifacts/timestamp/data_validation/validation_report.yaml
            # We need: artifacts/timestamp/data_ingestion/train.csv
            artifact_dir =os.path.dirname(os.path.dirname(data_validation_artifact.validation_report_file_path))
            self.train_file_path = os.path.join(artifact_dir, 'data_ingestion', 'train.csv')
            self.test_file_path = os.path.join(artifact_dir,'data_ingestion','test.csv')

            logging.info("Data Transformation component initialized")
            logging.info(f"Train file path: {self.train_file_path}")
            logging.info(f"Test file path: {self.test_file_path}")

        except Exception as e:
            raise BankChurnException(e, sys)

    def drop_unnecessary_columns(self, dataframe:pd.DataFrame)-> pd.DataFrame:
        """
        Drop columns that don't contribute to prediction.
        
        Why drop these columns?
        - RowNumber: Just an index, no predictive value
        - CustomerId: Unique identifier, causes overfitting
        - Surname: Too many unique values, no pattern
        
        Args:
            dataframe: Input DataFrame
        
        Returns:
            DataFrame with unnecessary columns removed
        """
        try:
            columns_to_drop = self.data_transformation_config.columns_to_drop
            logging.info(f"Dropping columns: {columns_to_drop}")
            # Check which columns exist
            existing_cols_to_drop = [
                col for col in columns_to_drop if col in dataframe.columns
                ]

            if existing_cols_to_drop:
                dataframe = dataframe.drop(columns=existing_cols_to_drop)
                logging.info(f"Dropped {len(existing_cols_to_drop)} columns")
            else:
                logging.info("No columns to drop (already removed)")

            return dataframe

        except Exception as e:
            logging.error(f"Error dropping columns: {str(e)}")
            raise BankChurnException(e, sys)

    def feature_engineering(self, dataframe:pd.DataFrame)->pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Based on your research:
        1. Log transformation for Age (improves distribution)
        2. Feature interactions (optional, can be extended)
        
        Why log(Age)?
        - Age has slight right skew
        - Log transformation normalizes distribution
        - Helps tree-based models find better splits
        - Tested and proven effective in your experiments
        
        Args:
            dataframe: Input DataFrame
        
        Returns:
            DataFrame with engineered features
        """

        try:
            logging.info('Starting Feature Engineering...')
            
            # 1.Age column transformation-Log
            if 'Age' in dataframe.columns:
                # add small constant to avoid log(0) if age=0 exists
                dataframe['Age_log'] = np.log1p(dataframe['Age'])
                logging.info('Created age_log feature')

            # 2. balance -to salary ration (financial health indicator)
            if 'Balance' in dataframe.columns and 'EstimatedSalary' in dataframe.columns:
                # avoid division by zero
                dataframe['Balance_Salary_Ratio']=(
                    dataframe["Balance"] / (dataframe['EstimatedSalary'] +1)
                )
                logging.info('Created Balance-Salary ratio feature')
            
            # 3. products per tenure(engaement rate)
            if 'NumOfProducts' in dataframe.columns and 'Tenure' in dataframe.columns:
                # avoid division by zero
                dataframe['Products_Per_Tenure']= (
                    dataframe['NumOfProducts'] / (dataframe['Tenure']+1)
                )
                logging.info('Created Products_Per_Tenure feature')

            # 4. Active Member with credit card(engangement indicator)
            if 'IsActiveMember' in dataframe.columns and 'HasCrCard' in dataframe.columns:
                dataframe['Active_CreditCard']= (
                    dataframe['IsActiveMember'] * dataframe['HasCrCard']
                )
                logging.info("Created Active_CreditCard feature")

            logging.info(f"Feature engineering completed. New shape: {dataframe.shape}")
            
            return dataframe
        except Exception as e:
            raise BankChurnException(e,sys)
    
    def get_data_transformer_object(self)-> ColumnTransformer:
        """
        Create preprocessing pipeline using ColumnTransformer.
        
        Pipeline Architecture:
        1. Numerical features → StandardScaler
        2. Categorical features → OneHotEncoder
        3. Combined using ColumnTransformer
        
        Why this approach?
        - Clean separation of numerical and categorical preprocessing
        - Prevents data leakage (scaler fit only on train)
        - Production-ready (single object for inference)
        - Handles mixed data types elegantly
        
        Returns:
            Fitted ColumnTransformer object
        
        Raises:
            BankChurnException: If pipeline creation fails
        """

        try:
            logging.info('Creating data transformer object')

            # define feature columns(after feature engg)
            numerical_features = [
                'CreditScore',
                'Age',
                'Age_log',  # Engineered feature
                'Tenure',
                'Balance',
                'Balance_Salary_Ratio',  # Engineered feature
                'NumOfProducts',
                'Products_Per_Tenure',  # Engineered feature
                'EstimatedSalary'
            ]

            categorical_features = ['Geography', 'Gender']

            binary_features = [
                'HasCrCard',
                'IsActiveMember',
                'Active_CreditCard'  # Engineered feature
            ]

            logging.info(f"Numerical features: {len(numerical_features)}")
            logging.info(f"Categorical features: {len(categorical_features)}")
            logging.info(f"Binary features: {len(binary_features)}")

            # Create preprocessing pipelines

            # Numerical pipeline: StandardScaler
            # Why StandardScaler?
            # - Normalizes features to mean=0, std=1
            # - Important for distance-based algorithms
            # - Less critical for tree-based models but doesn't hurt
            numerical_pipeline= Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline: OneHotEncoder
            # Why OneHotEncoder?
            # - Converts categories to binary columns
            # - handle_unknown='ignore' prevents errors with new categories
            # - sparse_output=False for compatibility with numpy arrays
            categorical_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ))
            ]) 

            # Binary features: Pass through (already 0/1)
            # We could scale them, but they're already in [0, 1] range

            # Combine all transformers

            preprocessor=ColumnTransformer(transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features),
                ('bin', 'passthrough', binary_features)
            ],
            remainder='drop' #  Drop any remaining columns
            )
            logging.info("Data transformer object created successfully")
            return preprocessor

        except Exception as e:
            logging.error(f"Error creating transformer object: {str(e)}")
            raise BankChurnException(e, sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Main method to orchestrate data transformation pipeline.
        
        Transformation Pipeline:
        1. Load train and test data
        2. Drop unnecessary columns
        3. Apply feature engineering
        4. Separate features and target
        5. Create and fit preprocessor (on train only!)
        6. Transform train and test data
        7. Combine features with target
        8. Save transformed arrays
        9. Save preprocessor object
        10. Return artifact
        
        Returns:
            DataTransformationArtifact with paths to transformed data and preprocessor
        
        Raises:
            BankChurnException: If transformation fails
        """ 
        try:
            logging.info("=" * 70)
            logging.info("STARTING DATA TRANSFORMATION")
            logging.info("=" * 70)

            # Step 1: Load data
            logging.info("\nStep 1: Loading train and test data...")

            train_df = pd.read_csv(self.train_file_path)
            test_df = pd.read_csv(self.test_file_path)
            
            logging.info(f"Train data loaded: {train_df.shape}")
            logging.info(f"Test data loaded: {test_df.shape}")

            # Step2: Drop unnecssary columns
            logging.info("\nStep 2: Dropping unnecessary columns...")
            train_df = self.drop_unnecessary_columns(train_df)
            test_df = self.drop_unnecessary_columns(test_df)

            logging.info(f"Train data after dropping: {train_df.shape}")
            logging.info(f"Test data after dropping: {test_df.shape}")

            # Step 3: Feature engineering
            logging.info("\nStep 3: Applying feature engineering...")
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            logging.info(f"Train data after engineering: {train_df.shape}")
            logging.info(f"Test data after engineering: {test_df.shape}")

            # Step 4: Separate features and target
            logging.info("\nStep 4: Separating features and target...")
            target_column = self.data_transformation_config.target_column

            # Feature(X)
            X_train = train_df.drop(columns=[target_column])
            X_test= test_df.drop(columns=[target_column])

            # Feature(y)
            y_train = train_df[target_column]
            y_test = test_df[target_column]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

            # Step 5: Create preprocessor
            logging.info("\nStep 5: Creating preprocessing pipeline...")
            preprocessor = self.get_data_transformer_object()
            
            # Step 6: Fit and transform
            logging.info("\nStep 6: Fitting preprocessor on training data...")
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info("Preprocessor fitted successfully")

            logging.info("Transforming test data...")
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Test data transformed successfully")

            logging.info(f"X_train_transformed shape: {X_train_transformed.shape}")
            logging.info(f"X_test_transformed shape: {X_test_transformed.shape}")

            # Step 7: Combine features with target
            logging.info("\nStep 7: Combining features with target...")
            train_arr= np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

            # Step 8: Save transformed arrays
            logging.info("\nStep 8: Saving transformed arrays...")
            save_numpy_array(file_path=self.data_transformation_config.transformed_train_file_path,
                             array=train_arr)
            logging.info(f"Train array saved to: {self.data_transformation_config.transformed_train_file_path}")

            save_numpy_array(file_path=self.data_transformation_config.transformed_test_file_path,
                             array=test_arr)
            logging.info(f"Test array saved to: {self.data_transformation_config.transformed_test_file_path}")

            # Step 9: Save preprocessor object
            logging.info("\nStep 9: Saving preprocessor object...")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor)
            logging.info(f"Preprocessor saved to: {self.data_transformation_config.preprocessor_obj_file_path}")

            # Step 10: Create artifact
            logging.info("\nStep 10: Creating data transformation artifact...")

            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_obj_file_path
            )
            logging.info("=" * 70)
            logging.info("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            logging.info("=" * 70)
            logging.info(f"Transformed train: {data_transformation_artifact.transformed_train_file_path}")
            logging.info(f"Transformed test: {data_transformation_artifact.transformed_test_file_path}")
            logging.info(f"Preprocessor: {data_transformation_artifact.preprocessor_object_file_path}")
            logging.info("=" * 70)
            return data_transformation_artifact

        except Exception as e:
            raise BankChurnException(e,sys)