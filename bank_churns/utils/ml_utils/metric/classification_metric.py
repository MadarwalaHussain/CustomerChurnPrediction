"""
Classification metrics calculator for the Bank Churn prediction system.
Provides comprehensive evaluation metrics for binary classification.
"""

import sys
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from bank_churns.exception.exception import BankChurnException
from bank_churns.entity.artifact_entity import ClassificationMetricArtifact
from bank_churns.logging.logger import logging

def get_classification_score(
        y_true: np.ndarray,
        y_pred: np.ndarray
)->ClassificationMetricArtifact:
    """
    Calculate comprehensive classification metrics for model evaluation.
    
    Why these specific metrics for churn prediction?
    
    1. **Accuracy**: Overall correctness (but can be misleading with imbalanced data)
    2. **Precision**: Of customers we predicted would churn, how many actually did?
       - High precision = Few false alarms (don't waste marketing budget)
    3. **Recall**: Of customers who actually churned, how many did we catch?
       - High recall = Catch most churners (prevent revenue loss)
    4. **F1-Score**: Harmonic mean of precision and recall
       - Best single metric for imbalanced datasets
       - Balances false positives and false negatives
    5. **ROC-AUC**: Model's ability to distinguish between classes
       - 0.5 = Random guessing
       - 1.0 = Perfect classification
       - >0.85 is considered good for production
    
    Args:
        y_true: Ground truth labels (0 or 1)
                0 = Customer stayed
                1 = Customer churned
        y_pred: Predicted labels (0 or 1)
    
    Returns:
        ClassificationMetricArtifact containing all metrics
    
    Raises:
        BankChurnException: If metric calculation fails
    
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        >>> y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
        >>> metrics = get_classification_score(y_true, y_pred)
        >>> print(f"F1 Score: {metrics.f1_score:.3f}")
        >>> print(f"Precision: {metrics.precision_score:.3f}")
        >>> print(f"Recall: {metrics.recall_score:.3f}")
    """
    try:
        logging.info('Calculting classification metrics')

        # Valid Inputs
        if len(y_true)!=len(y_pred):
            raise ValueError(
                f'Length mismatch: y_true has {len(y_true)} samples,'
                f'y_pred has {len(y_pred)}samples'
            )
        
        if len(y_true)==0:
            raise ValueError('Cannot calculate metrics for empty arrays')
        
        # Calculate Metrics:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true,y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1=f1_score(y_true, y_pred, zero_division=0)

            # ROC AUC(requires atleast one sample of each class)
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except ValueError as e:
            # if only one class present in y_true, ROC-AUC is undefined
            logging.warning(f'Could not calculate ROC_AUC: {str(e)}')
            roc_auc=0
    
        # Log metrics
        logging.info("=" * 60)
        logging.info("CLASSIFICATION METRICS")
        logging.info("=" * 60)
        logging.info(f"Accuracy:     {accuracy:.4f}")
        logging.info(f"Precision:    {precision:.4f}")
        logging.info(f"Recall:       {recall:.4f}")
        logging.info(f"F1-Score:     {f1:.4f}")
        logging.info(f"ROC-AUC:      {roc_auc:.4f}")
        logging.info("=" * 60)

        # Calculate and log confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logging.info("\nConfusion Matrix:")
        logging.info(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
        logging.info(f"FN: {cm[1][0]}, TP: {cm[1][1]}")

        # Create and return artifact
        classification_metric = ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall,
            accuracy_score=accuracy,
            roc_auc_score=roc_auc
        )

        logging.info('Classification metrics calculated successfully')
        return classification_metric
    
    except Exception as e:
        logging.error(f"Error calculating classification metrics: {str(e)}")
        raise BankChurnException(e, sys)


def log_detailed_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Log detailed classification report with per-class metrics.
    
    This provides more granular insights than aggregate metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Example output:
                  precision    recall  f1-score   support
        
               0       0.88      0.95      0.91      1607
               1       0.73      0.52      0.61       393
        
        accuracy                           0.86      2000
       macro avg       0.81      0.73      0.76      2000
    weighted avg       0.85      0.86      0.85      2000
    """
    try:
        report = classification_report(
            y_true,
            y_pred,
            target_names=['Stayed (0)', 'Churned (1)'],
            digits=4
        )

        logging.info("\n" + "=" * 60)
        logging.info("DETAILED CLASSIFICATION REPORT")
        logging.info("=" * 60)
        logging.info("\n" + report)
        logging.info("=" * 60)

        return report
    except Exception as e:
        logging.warning(f"Could not generate detailed report: {str(e)}")


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate business-relevant metrics for churn prediction.
    
    These metrics help stakeholders understand the business impact:
    - How many churners did we miss? (False Negatives = Lost customers)
    - How many false alarms? (False Positives = Wasted marketing budget)
    - What's our true positive rate? (Successfully identified churners)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with business metrics
    
    Example:
        >>> metrics = calculate_business_metrics(y_true, y_pred)
        >>> print(f"We missed {metrics['false_negatives']} churners")
        >>> print(f"We had {metrics['false_positives']} false alarms")
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate rates
        total = len(y_true)
        actual_churn_count = np.sum(y_true == 1)
        predicted_churn_count = np.sum(y_pred == 1)

        business_metrics = {
            # Confusion matrix values
            'true_negatives': int(tn),      # Correctly predicted as staying
            'false_positives': int(fp),     # False alarms (predicted churn, but stayed)
            'false_negatives': int(fn),     # Missed churners (predicted stay, but churned)
            'true_positives': int(tp),      # Correctly predicted as churning

            # Rates
            'actual_churn_rate': float(actual_churn_count / total),
            'predicted_churn_rate': float(predicted_churn_count / total),

            # Business KPIs
            'churn_capture_rate': float(tp / actual_churn_count) if actual_churn_count > 0 else 0.0,
            'false_alarm_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'missed_churn_rate': float(fn / actual_churn_count) if actual_churn_count > 0 else 0.0
        }

        # Log business metrics
        logging.info("\n" + "=" * 60)
        logging.info("BUSINESS METRICS")
        logging.info("=" * 60)
        logging.info(f"Actual churn rate:     {business_metrics['actual_churn_rate']:.2%}")
        logging.info(f"Predicted churn rate:  {business_metrics['predicted_churn_rate']:.2%}")
        logging.info(f"Churn capture rate:    {business_metrics['churn_capture_rate']:.2%}")
        logging.info(f"False alarm rate:      {business_metrics['false_alarm_rate']:.2%}")
        logging.info(f"Missed churners:       {business_metrics['false_negatives']} customers")
        logging.info("=" * 60)

        return business_metrics

    except Exception as e:
        logging.error(f"Error calculating business metrics: {str(e)}")
        raise BankChurnException(e, sys)


def compare_model_metrics(
        train_metric: ClassificationMetricArtifact,
        test_metric: ClassificationMetricArtifact,
        threshold: float=0.5
)->dict:
    """
    Compare train and test metrics to detect overfitting/underfitting.
    
    Args:
        train_metric: Metrics from training set
        test_metric: Metrics from test set
        threshold: Maximum acceptable difference (default 5%)
    
    Returns:
        Dictionary with comparison results and overfitting status
    
    Example:
        >>> result = compare_model_metrics(train_metrics, test_metrics)
        >>> if result['is_overfitting']:
        ...     print("Model is overfitting - retrain with regularization")
    
    """
    try:
        # calculate differences
        f1_diff= abs(train_metric.f1_score - test_metric.f1_score)
        precision_diff=abs(train_metric.precision_score - test_metric.precision_score)
        recall_diff = abs(train_metric.recall_score - test_metric.recall_score)
        accuracy_diff= abs(train_metric.accuracy_score - test_metric.accuracy_score)

        # check for overfitting
        is_overfitting = (
            f1_diff > threshold or
            accuracy_diff > threshold
        )

        comparison={
            'train_f1': train_metric.f1_score,
            'test_f1': test_metric.f1_score,
            'f1_difference': f1_diff,
            'train_accuracy': train_metric.accuracy_score,
            'test_accuracy': test_metric.accuracy_score,
            'accuracy_difference': accuracy_diff,
            'is_overfitting': is_overfitting,
            'threshold': threshold
        }
        # Log comparison
        logging.info("\n" + "=" * 60)
        logging.info("TRAIN vs TEST COMPARISON")
        logging.info("=" * 60)
        logging.info(f"Train F1:       {train_metric.f1_score:.4f}")
        logging.info(f"Test F1:        {test_metric.f1_score:.4f}")
        logging.info(f"F1 Difference:  {f1_diff:.4f}")
        logging.info(f"Train Accuracy: {train_metric.accuracy_score:.4f}")
        logging.info(f"Test Accuracy:  {test_metric.accuracy_score:.4f}")
        logging.info(f"Acc Difference: {accuracy_diff:.4f}")
        logging.info(f"Overfitting:    {'YES' if is_overfitting else 'NO'}")
        logging.info("=" * 60)

        return comparison

    except Exception as e:
        logging.error(f"Error comparing model metrics: {str(e)}")
        raise BankChurnException(e, sys)
