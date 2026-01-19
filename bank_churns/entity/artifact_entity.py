"""
Artifact entity classes for the Bank Churn prediction pipeline.
Stores outputs from each pipeline component for downstream use.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class DataIngestionArtifact:
    """
    Output artifact from data ingestion component.
    Contains paths to raw, train, and test datasets.
    """
    trained_file_path: str
    test_file_path: str
    raw_data_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Output artifact from data validation component.
    Contains validation status and report paths.
    """
    validation_status: bool
    validation_report_file_path: str
    drift_report_file_path: Optional[str] = None
    message: str = ""


@dataclass
class DataTransformationArtifact:
    """
    Output artifact from data transformation component.
    Contains paths to transformed data and preprocessor object.
    """
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str


@dataclass
class ClassificationMetricArtifact:
    """
    Classification metrics for model evaluation.
    """
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy_score: float
    roc_auc_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Output artifact from model trainer component.
    Contains trained model path and performance metrics.
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
    model_name: str = "RandomForestClassifier"


@dataclass
class ModelEvaluationArtifact:
    """
    Output artifact from model evaluation component.
    Determines if new model should be deployed.
    """
    is_model_accepted: bool
    improved_score: float
    current_model_score: float
    new_model_score: float
    evaluation_report_file_path: str


@dataclass
class ModelPusherArtifact:
    """
    Output artifact from model pusher component.
    Contains paths to deployed production models.
    """
    saved_model_path: str
    saved_preprocessor_path: str
    deployment_timestamp: str
