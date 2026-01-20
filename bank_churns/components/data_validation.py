"""
Data Validation Component for Bank Churn Prediction System.
Validates data quality, schema compliance, and detects drift.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import ks_2samp

from bank_churns.entity.config_entity import DataValidationConfig
from bank_churns.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from bank_churns.logging.logger import logging
from bank_churns.exception.exception import BankChurnException

from bank_churns.utils.main_utils.utils import write_yaml_file


class DataValidation:
    """
    Data Validation Component - Quality gates before transformation.

    Responsibilities:
    1. Validate schema(columns, data types)
    2. Check data quality(missing values, duplicates)
    3. Detect data drift(train vs test distributions)
    4. Generate validation reports

    Why validation matters:
    - Prevents garbage data from entering pipeline
    - Catches data collection errors early
    - Ensures reproducibility
    - Detects distribution shifts

    Example:
        >> > validation = DataValidation(
        ...     data_ingestion_artifact,
        ...     data_validation_config
        ...)
        >> > artifact = validation.initiate_data_validation()
        >> > if artifact.validation_status:
        ... print("Data is valid, proceed to transformation")
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initialize Data Validation component.

        Args:
            data_ingestion_artifact: Output from data ingestion (file paths)
            data_validation_config: Configuration for validation

        Raises:
            BankChurnException: If initialization fails
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            # Load train and test data
            self.train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            self.test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Data Validation component initialized")
            logging.info(f"Train data shape: {self.train_df.shape}")
            logging.info(f"Test data shape: {self.test_df.shape}")

        except Exception as e:
            raise BankChurnException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that dataframe has the expected number of columns.

        Why this matters:
        - Missing columns = incomplete data
        - Extra columns = data collection changed
        - Both indicate problems upstream

        Args:
            dataframe: DataFrame to validate

        Returns:
            True if column count matches, False otherwise
        """

        try:
            expected_columns = self.data_validation_config.expected_columns
            target_columns = self.data_validation_config.target_column

            expected_count = len(expected_columns) + 1
            actual_count = len(dataframe.columns)

            # we need to have extra columns to drop(RowNumber, CustomerId, Surname)
            # so we just check if expected columns exist
            missing_columns = []
            for col in expected_columns:
                if col not in dataframe.columns:
                    missing_columns.append(col)

            if target_columns not in dataframe.columns:
                missing_columns.append(target_columns)

            if missing_columns:
                logging.error(f"Missing Columns: {missing_columns}")
                return False

            logging.info(f'Columns validation passed . COlumns: {actual_count}')
            return True

        except Exception as e:
            logging.error(f"Error validating columns: {str(e)}")
            raise BankChurnException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that all expected numerical columns exist.

        Args:
            dataframe: DataFrame to validate

        Returns:
            True if all numerical columns exist, False otherwise
        """

        try:
            numerical_columns = [
                'CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'EstimatedSalary'
            ]
            missing_numerical = []
            for col in numerical_columns:
                if col not in dataframe.columns:
                    missing_numerical.append(col)

            if missing_numerical:
                logging.error(f"Missing numerical columns: {missing_numerical}")
                return False

            logging.info("All numerical columns present")
            return True

        except Exception as e:
            raise BankChurnException(e, sys)

    def is_categorical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that all expected categorical columns exist.

        Args:
            dataframe: DataFrame to validate

        Returns:
            True if all categorical columns exist, False otherwise
        """
        try:
            categorical_columns = ['Geography', 'Gender']

            missing_categorical = []
            for col in categorical_columns:
                if col not in dataframe.columns:
                    missing_categorical.append(col)

            if missing_categorical:
                logging.error(f"Missing categorical columns: {missing_categorical}")
                return False

            logging.info("All categorical columns present")
            return True

        except Exception as e:
            logging.error(f"Error checking categorical columns: {str(e)}")
            raise BankChurnException(e, sys)

    def check_data_quality(self, dataframe: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality checks.

        Checks:
        1. Missing values
        2. Duplicate rows
        3. Data types
        4. Value ranges
        5. Categorical value distributions

        Args:
            dataframe: DataFrame to check

        Returns:
            Dictionary with quality check results
        """

        try:
            logging.info('starting data quality checks..')
            quality_report = {}

            # 1.Missing values
            missing_values = dataframe.isna().sum()
            missing_percentage = (missing_values/len(dataframe)) * 100

            quality_report['missing_values'] = {
                col: {
                    'count': int(missing_values[col]),
                    'percentage': float(missing_percentage[col])

                }
                for col in dataframe.columns if missing_values[col] > 0
            }

            total_missing = missing_values.sum()
            logging.info(f'Total missing values:{total_missing}')

            # 2.Duplicate rows
            duplicate_count = dataframe.duplicated().sum()
            quality_report['duplicate_rows'] = int(duplicate_count)
            logging.info(f'Duplicate rows: {duplicate_count}')

            # 3.Data Types
            quality_report['data_types'] = {
                col: str(dtype) for col, dtype in dataframe.dtypes.items()
            }

            # 4. Basic Statistics for numerical columns
            numerical_cols = dataframe.select_dtypes(include=[np.number]).columns
            quality_report['numerical_stats'] = {}

            for col in numerical_cols:
                quality_report['numerical_stats'][col] = {
                    'min': float(dataframe[col].min()),
                    'max': float(dataframe[col].max()),
                    'mean': float(dataframe[col].mean()),
                    'std': float(dataframe[col].std())
                }

            # 5.Categorical stats
            categorical_cols = ['Geography', 'Gender']
            quality_report['categorical_distributions'] = {}

            for col in categorical_cols:
                if col in dataframe.columns:
                    value_counts = dataframe[col].value_counts().to_dict()
                    quality_report['categorical_distributions'][col] = {
                        str(k): int(v) for k, v in value_counts.items()
                    }

            # 6. Target dist
            if 'Exited' in dataframe.columns:
                target_dist = dataframe['Exited'].value_counts().to_dict()
                quality_report['target_distribution'] = {
                    str(k): int(v) for k, v in target_dist.items()
                }
                churn_rate = (dataframe['Exited'].sum()/len(dataframe)) * 100
                quality_report['churn_rate'] = float(churn_rate)
                logging.info('Data Quality check completed')
                return quality_report

        except Exception as e:
            raise BankChurnException(e, sys)

    def detect_dataset_drift(self,
                             base_df: pd.DataFrame,
                             current_df: pd.DataFrame,
                             threshold: float = 0.05) -> Tuple[bool, Dict]:
        """
            Detect data drift between train and test sets using KS test.

        Why drift detection matters:
        - Train and test should come from same distribution
        - Significant drift indicates data collection issues
        - Or indicates temporal shift (different time periods)

        Kolmogorov-Smirnov Test:
        - Compares two distributions
        - p-value < 0.05 = distributions are different
        - Tests each numerical feature independently

        Args:
            base_df: Base dataset (usually training data)
            current_df: Current dataset (usually test data)
            threshold: p-value threshold (default 0.05)

        Returns:
            Tuple of (drift_detected: bool, drift_report: dict)

        Example:
            drift_detected, report = detect_dataset_drift(train_df, test_df)
            if drift_detected:
                print(f"Drift detected in: {report['drifted_columns']}")
            """
        try:
            logging.info('starting data drift detection...')
            drift_report = {
                'drift_detected': False,
                'drifted_columns': [],
                'drift_scores': {},
                'threshold': threshold
            }

            # get numerical cols
            numerical_columns = base_df.select_dtypes(include=[np.number]).columns

            for column in numerical_columns:
                ks_statistic, p_value = ks_2samp(
                    base_df[column],
                    current_df[column]
                )
                drift_report['drift_scores'][column] = {
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value),
                    'drift_detected': bool(p_value < threshold)
                }

                # check if drift detected
                if p_value < threshold:
                    drift_report['drifted_columns'].append(column)
                    drift_report['drift_detected'] = True
                    logging.warning(
                        f"Drift detected in '{column}': "
                        f"p_value={p_value:.4f} < {threshold}"
                    )
                else:
                    logging.info(
                        f"No drift in '{column}': p_value={p_value:.4f}"
                    )
            if drift_report['drift_detected']:
                logging.warning(
                    f"Data drift detected in {len(drift_report['drifted_columns'])} columns"
                )
            else:
                logging.info("No significant data drift detected")

            return bool(drift_report['drift_detected']), drift_report
        except Exception as e:
            raise BankChurnException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Main method to orchestrate all validation checks.

        Validation Pipeline:
        1. Schema validation (columns exist)
        2. Column type validation (numerical, categorical)
        3. Data quality checks (missing, duplicates, stats)
        4. Data drift detection (train vs test)
        5. Generate validation report
        6. Return artifact with status

        Returns:
            DataValidationArtifact with validation status and report paths

        Raises:
            BankChurnException: If validation process fails
        """
        try:
            logging.info("=" * 70)
            logging.info("STARTING DATA VALIDATION")
            logging.info("=" * 70)

            validation_error_messages = []

            # step1: validate number of columns(Train)
            logging.info('Validating number of columns')
            train_columns_valid = self.validate_number_of_columns(self.train_df)
            if not train_columns_valid:
                validation_error_messages.append("Train data: Column validation failed")

            test_columns_valid = self.validate_number_of_columns(self.test_df)
            if not test_columns_valid:
                validation_error_messages.append("Test data: Column validation failed")

            # Step 2: Validate numerical columns
            train_numerical_valid = self.is_numerical_column_exist(self.train_df)
            test_numerical_valid = self.is_numerical_column_exist(self.test_df)

            if not train_numerical_valid:
                validation_error_messages.append("Train data: Missing numerical columns")
            if not test_numerical_valid:
                validation_error_messages.append("Test data: Missing numerical columns")

            # Step 3: Validate categorical columns
            logging.info("\nStep 3: Validating categorical columns...")
            train_categorical_valid = self.is_categorical_column_exist(self.train_df)
            test_categorical_valid = self.is_categorical_column_exist(self.test_df)

            if not train_categorical_valid:
                validation_error_messages.append("Train data: Missing categorical columns")
            if not test_categorical_valid:
                validation_error_messages.append("Test data: Missing categorical columns")

            # Step 4: Data quality checks
            logging.info("\nStep 4: Performing data quality checks...")
            train_quality_report = self.check_data_quality(self.train_df)
            test_quality_report = self.check_data_quality(self.test_df)

            # Step 5: Drift detection
            logging.info("\nStep 5: Detecting data drift...")
            drift_detected, drift_report = self.detect_dataset_drift(
                self.train_df,
                self.test_df
            )
            # Compile validation report
            validation_report = {
                'validation_status': len(validation_error_messages) == 0,
                'error_messages': validation_error_messages,
                'train_data': {
                    'shape': list(self.train_df.shape),
                    'columns_valid': train_columns_valid,
                    'numerical_valid': train_numerical_valid,
                    'categorical_valid': train_categorical_valid,
                    'quality': train_quality_report
                },
                'test_data': {
                    'shape': list(self.test_df.shape),
                    'columns_valid': test_columns_valid,
                    'numerical_valid': test_numerical_valid,
                    'categorical_valid': test_categorical_valid,
                    'quality': test_quality_report
                }
            }

            # Save validation report
            logging.info('Saving validation report....')
            write_yaml_file(self.data_validation_config.validation_report_file_path, validation_report)

            # save drift report
            logging.info('Saving drift report....')
            write_yaml_file(self.data_validation_config.drift_report_file_path, drift_report)

            # Determine overall validation status
            validation_status = len(validation_error_messages) == 0

            # Create message
            if validation_status:
                message = "All validation checks passed"
                logging.info(message)
            else:
                message = f"Validation failed: {', '.join(validation_error_messages)}"
                logging.error(message)

            # create artifacts
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                validation_report_file_path=self.data_validation_config.validation_report_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                message=message
            )
            logging.info("=" * 70)
            logging.info("DATA VALIDATION COMPLETED")
            logging.info("=" * 70)
            logging.info(f"Status: {'PASSED' if validation_status else 'FAILED'}")
            logging.info(f"Validation report: {data_validation_artifact.validation_report_file_path}")
            logging.info(f"Drift report: {data_validation_artifact.drift_report_file_path}")
            logging.info("=" * 70)

            return data_validation_artifact
        except Exception as e:
            raise BankChurnException(e, sys)
