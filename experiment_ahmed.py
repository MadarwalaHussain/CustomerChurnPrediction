"""
Ahmed's Experiments - Linear Models & SVM
Week 1: Baseline with Logistic Regression and SVM
"""

from bank_churns.utils.main_utils.utils import load_numpy_array
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import dagshub
import mlflow
import sys
import os

# STEP 1: Load environment FIRST
from dotenv import load_dotenv
load_dotenv()

# STEP 2: Apply SSL fix BEFORE mlflow
try:
    import fix_ssl
    fix_ssl.apply_ssl_fix()
    print("SSL fix applied")
except ImportError:
    print("fix_ssl not found")
except Exception as e:
    print(f"SSL fix failed: {e}")

# STEP 3: SSL environment variables
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# STEP 4: NOW import mlflow/dagshub

# STEP 5: Other imports

# Get credentials
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_user_name = os.getenv('MLFLOW_TRACKING_USERNAME')

# Initialize DagsHub
if mlflow_uri and mlflow_user_name:
    try:
        dagshub.init(
            repo_owner='MadarwalaHussain',
            repo_name='CustomerChurnPrediction',
            mlflow=True
        )
        print("✓ DagsHub initialized")
    except Exception as e:
        print(f"⚠ DagsHub init failed: {e}")

# Setup MLflow
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("churn-prediction-team")

# Load data (UPDATE THIS PATH TO YOUR ACTUAL PATH)
ARTIFACT_PATH = r"C:\Users\hmadarw\Learnings\DataScience\Projects\CustomerChurnPrediction\artifacts\20260122_084136\data_transformation\transformed"

train_arr = load_numpy_array(os.path.join(ARTIFACT_PATH, "train.npy"))
test_arr = load_numpy_array(os.path.join(ARTIFACT_PATH, "test.npy"))

X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

print("=" * 70)
print("AHMED'S EXPERIMENTS - Linear & SVM Models")
print("=" * 70)
print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# ============================================
# EXPERIMENT 1: Logistic Regression - Baseline
# ============================================
print("\n[1/3] Logistic Regression - Baseline...")
with mlflow.start_run(run_name="ahmed_lr_baseline"):
    # Tags
    mlflow.set_tag("engineer", "ahmed")
    mlflow.set_tag("model_type", "logistic_regression")
    mlflow.set_tag("experiment_phase", "week1_baseline")

    # Train
    lr_model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    # Evaluate
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log params
    mlflow.log_param("C", 1.0)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)

    # Log metrics
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(lr_model, "model")

    print(f" F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================
# EXPERIMENT 2: Logistic Regression - L2 Regularized
# ============================================
print("\n[2/3] Logistic Regression - L2 Regularized...")
with mlflow.start_run(run_name="ahmed_lr_regularized"):
    mlflow.set_tag("engineer", "ahmed")
    mlflow.set_tag("model_type", "logistic_regression")
    mlflow.set_tag("experiment_phase", "week1_optimized")

    lr_reg = LogisticRegression(
        C=0.1,  # Stronger regularization
        penalty='l2',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_reg.fit(X_train, y_train)

    y_pred = lr_reg.predict(X_test)
    y_pred_proba = lr_reg.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("C", 0.1)
    mlflow.log_param("penalty", "l2")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(lr_reg, "model")

    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================
# EXPERIMENT 3: SVM - RBF Kernel
# ============================================
print("\n[3/3] SVM - RBF Kernel...")
print("(Using 2000 samples for speed - SVM is slow on large datasets)")
with mlflow.start_run(run_name="ahmed_svm_rbf"):
    mlflow.set_tag("engineer", "ahmed")
    mlflow.set_tag("model_type", "svm")
    mlflow.set_tag("experiment_phase", "week1_baseline")

    # SVM is slow, use subset for demo
    subset_size = 2000
    X_train_sub = X_train[:subset_size]
    y_train_sub = y_train[:subset_size]

    svm_model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_sub, y_train_sub)

    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("C", 1.0)
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("gamma", "scale")
    mlflow.log_param("training_subset_size", subset_size)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(svm_model, "model")

    print(f" F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

print("\n" + "=" * 70)
print("AHMED'S EXPERIMENTS COMPLETED")
print("=" * 70)
print("3 experiments logged to MLflow")
print( "Check DagsHub: https://dagshub.com/MadarwalaHussain/CustomerChurnPrediction/experiments")
print("=" * 70)
