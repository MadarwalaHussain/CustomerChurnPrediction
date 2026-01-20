"""
Data Ingestion Component for Bank Churn Prediction System.
Handles loading CSV data, train-test split, and artifact creation.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

from bank_churns.constants.training_pipeline import TARGET_COLUMN
from bank_churns.entity.config_entity import DataIngestionConfig
from bank_churns.entity.artifact_entity import DataIngestionArtifact
from bank_churns.logging.logger import logging
from bank_churns.exception.exception import BankChurnException


class DataIngestion:
    """
    Data Ingestion Component - First step in ML pipeline.

    Responsibilities:
    1. Load raw CSV data
    2. Validate data exists and is readable
    3. Save raw data to artifact directory
    4. Perform stratified train-test split
    5. Save train and test sets
    6. Return artifact with file paths

    Why stratified split?
    - Preserves class distribution in train/test
    - Critical for imbalanced data (20% churn rate)
    - Ensures both sets have similar churn rates

    Example:
        >>> config = DataIngestionConfig(training_pipeline_config)
        >>> data_ingestion = DataIngestion(config)
        >>> artifact = data_ingestion.initiate_data_ingestion()
        >>> print(artifact.trained_file_path)
        'artifacts/20250119_120530/data_ingestion/train.csv'
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize Data Ingestion component.

        Args:
            data_ingestion_config: Configuration with paths and parameters

        Raises:
            BankChurnException: If initialization fails
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("Data Ingestion component initialized")

        except Exception as e:
            raise BankChurnException(e, sys)

    def load_data_from_csv(self) -> pd.DataFrame:
        """
        Load data from source CSV file.

        Returns:
            DataFrame with raw bank churn data

        Raises:
            BankChurnException: If file not found or read fails

        Notes:
            - Validates file exists before reading
            - Logs data shape and basic info
            - Checks for completely empty DataFrame
        """
        try:
            source_path = self.data_ingestion_config.source_data_path

            # validate file exists
            if not os.path.exists(source_path):
                raise FileNotFoundError(
                    f"source data file not found: {source_path}\n"
                    f"please ensure 'churn.csv exits in the project roor directory"
                )

            logging.info(f"Loading data from: {source_path}")

            # Read csv
            dataframe = pd.read_csv(source_path)

            # validate data loaded
            if dataframe.empty:
                raise ValueError("Loaded Dataframe is empty")

            logging.info(f"Data loaded successfully. Shape: {dataframe.shape}")
            logging.info(f"Columns: {dataframe.columns.tolist()}")

            # check basic stats
            logging.info(f"Total records: {len(dataframe):,}")
            logging.info(f"Total features: {len(dataframe.columns)}")

            # check target column
            if TARGET_COLUMN in dataframe.columns:
                churn_count = dataframe[TARGET_COLUMN].sum()
                churn_rate = (churn_count/len(dataframe)) * 100
                logging.info(f'Churn rate: {churn_rate:.2f}% ({churn_count}/{len(dataframe)})')

            return dataframe

        except Exception as e:
            logging.error(f"Error loading data from CSV: {str(e)}")
            raise BankChurnException(e, sys)

    def save_raw_data(self, dataframe: pd.DataFrame) -> None:
        """
        Save complete raw dataset to artifact directory.

        Why save raw data?
        - Reproducibility: Can always trace back to original data
        - Debugging: Compare transformed data with raw
        - Audit trail: Know exactly what data was used
        - Data versioning: Each pipeline run has its own copy

        Args:
            dataframe: Raw data DataFrame to save

        Raises:
            BankChurnException: If saving fails
        """
        try:
            raw_file_path = self.data_ingestion_config.raw_data_file_path

            logging.info(f"Saving raw data to: {raw_file_path}")

            # ensure dirctory exists
            os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)

            # save to csv
            dataframe.to_csv(raw_file_path, index=False)

            # Log file info
            file_size = os.path.getsize(raw_file_path) / (1024 * 1024)  # MB
            logging.info(f"Raw data saved successfully. Size: {file_size:.2f} MB")

        except Exception as e:
            logging.error(f"Error saving raw data: {str(e)}")
            raise BankChurnException(e, sys)


    def split_data_as_train_test(self, dataframe:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets with stratification.
        
        Why stratified split?
        Example without stratification:
        - Overall churn rate: 20%
        - Train churn rate: 18% (unlucky split)
        - Test churn rate: 25% (unlucky split)
        - Model trained on 18% but tested on 25% --> misleading metrics!
        
        With stratification:
        - Overall churn rate: 20%
        - Train churn rate: ~20% (guaranteed)
        - Test churn rate: ~20% (guaranteed)
        - Fair and reproducible evaluation
        
        Args:
            dataframe: Complete dataset
        
        Returns:
            Tuple of (train_df, test_df)
        
        Raises:
            BankChurnException: If split fails
        """

        try:
            logging.info('Starting train-test split')
            
            # check if target column exits
            if TARGET_COLUMN not in dataframe.columns:
                raise ValueError(
                    f'Target column {TARGET_COLUMN} not found in data'
                    f'Available columns: {dataframe.columns.tolist()}'
                )
            
            # Performs stratified split
            train_set, test_set = train_test_split(dataframe,
                                                   test_size=self.data_ingestion_config.train_test_split_ratio,
                                                   random_state=self.data_ingestion_config.random_state,
                                                   stratify=dataframe[TARGET_COLUMN])
            
            logging.info(f"Train set size: {len(train_set):,} ({len(train_set)/len(dataframe)*100:.1f}%)")
            logging.info(f"Test set size: {len(test_set):,} ({len(test_set)/len(dataframe)*100:.1f}%)")
            
            # Verify stratification worked
            train_churn_rate = (train_set['Exited'].sum() / len(train_set)) * 100
            test_churn_rate = (test_set['Exited'].sum() / len(test_set)) * 100

            logging.info(f"Train churn rate: {train_churn_rate:.2f}%")
            logging.info(f"Test churn rate: {test_churn_rate:.2f}%")
            logging.info("Train-test split completed successfully")

            return train_set, test_set
        
        except Exception as e:
            raise BankChurnException(e,sys)
        
    def save_train_test_data(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """
        Save train and test sets to artifact directory.
        
        Args:
            train_set: Training data
            test_set: Testing data
        
        Raises:
            BankChurnException: If saving fails
        """
        try:
            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            # Save train set
            logging.info(f"Saving train data to: {train_file_path}")
            train_set.to_csv(train_file_path, index=False)
            train_size = os.path.getsize(train_file_path) / (1024 * 1024)
            logging.info(f"Train data saved. Size: {train_size:.2f} MB")

            # Save test set
            logging.info(f"Saving test data to: {test_file_path}")
            test_set.to_csv(test_file_path, index=False)
            test_size = os.path.getsize(test_file_path) / (1024 * 1024)
            logging.info(f"Test data saved. Size: {test_size:.2f} MB")

        except Exception as e:
            logging.error(f"Error saving train/test data: {str(e)}")
            raise BankChurnException(e, sys)
        

    def init_data_ingestion(self)-> DataIngestionArtifact:
        """
        Main method to orchestrate data ingestion pipeline.
        
        Pipeline steps:
        1. Load data from CSV
        2. Save raw data copy
        3. Split into train/test
        4. Save train/test sets
        5. Create and return artifact
        
        Returns:
            DataIngestionArtifact with paths to all saved files
        
        Raises:
            BankChurnException: If any step fails
        
        Example:
            >>> artifact = data_ingestion.initiate_data_ingestion()
            >>> print(artifact.trained_file_path)
            'artifacts/20250119_120530/data_ingestion/train.csv'
        """
        try:
            logging.info("=" * 70)
            logging.info("STARTING DATA INGESTION")
            logging.info("=" * 70)

            # Step 1: Load data from CSV
            logging.info("Step 1: Loading data from CSV")
            dataframe = self.load_data_from_csv()

            # Step 2: Save raw data
            logging.info("Step 2: Saving raw data")
            self.save_raw_data(dataframe)

            # Step 3: Split data
            logging.info("Step 3: Splitting data into train and test sets")
            train_set, test_set = self.split_data_as_train_test(dataframe)

            # Step 4: Save train and test
            logging.info("Step 4: Saving train and test data")
            self.save_train_test_data(train_set, test_set)

            # Step 5: Create artifact
            logging.info("Step 5: Creating data ingestion artifact")
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
                raw_data_file_path=self.data_ingestion_config.raw_data_file_path
            )

            logging.info("=" * 70)
            logging.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logging.info("=" * 70)
            logging.info(f"Train file: {data_ingestion_artifact.trained_file_path}")
            logging.info(f"Test file: {data_ingestion_artifact.test_file_path}")
            logging.info(f"Raw file: {data_ingestion_artifact.raw_data_file_path}")
            logging.info("=" * 70)

            return data_ingestion_artifact

        except Exception as e:
            logging.error("=" * 70)
            logging.error("DATA INGESTION FAILED")
            logging.error("=" * 70)
            raise BankChurnException(e, sys)
