"""
Hussain's Experiments - Tree-based Models
Week 1: Initial baseline with Random Forest and XGBoost
"""

from bank_churns.utils.main_utils.utils import load_numpy_array
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
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
    print("✓ SSL fix applied")
except ImportError:
    print("⚠ fix_ssl not found")
except Exception as e:
    print(f"⚠ SSL fix failed: {e}")

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
print("HUSSAIN'S EXPERIMENTS - Tree-Based Models")
print("=" * 70)
print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# ============================================
# EXPERIMENT 1: Random Forest - Baseline
# ============================================
print("\n[1/4] Random Forest - Baseline...")
with mlflow.start_run(run_name="hussain_rf_baseline"):
    # Tags
    mlflow.set_tag("engineer", "hussain")
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("experiment_phase", "week1_baseline")

    # Train
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log params
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")

    # Log metrics
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(rf_model, "model")

    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================
# EXPERIMENT 2: Random Forest - Optimized
# ============================================
print("\n[2/4] Random Forest - Optimized...")
with mlflow.start_run(run_name="hussain_rf_optimized"):
    mlflow.set_tag("engineer", "hussain")
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("experiment_phase", "week1_optimized")

    rf_opt = RandomForestClassifier(
        n_estimators=294,
        max_depth=11,
        min_samples_split=7,
        min_samples_leaf=9,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_opt.fit(X_train, y_train)

    y_pred = rf_opt.predict(X_test)
    y_pred_proba = rf_opt.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("n_estimators", 294)
    mlflow.log_param("max_depth", 11)
    mlflow.log_param("min_samples_split", 7)
    mlflow.log_param("min_samples_leaf", 9)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(rf_opt, "model")

    print(f" F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================
# EXPERIMENT 3: XGBoost - Baseline
# ============================================
print("\n[3/4] XGBoost - Baseline...")
with mlflow.start_run(run_name="hussain_xgb_baseline"):
    mlflow.set_tag("engineer", "hussain")
    mlflow.set_tag("model_type", "xgboost")
    mlflow.set_tag("experiment_phase", "week1_baseline")

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(xgb_model, "model")

    print(f" F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================
# EXPERIMENT 4: XGBoost - Optimized
# ============================================
print("\n[4/4] XGBoost - Optimized...")
with mlflow.start_run(run_name="hussain_xgb_optimized"):
    mlflow.set_tag("engineer", "hussain")
    mlflow.set_tag("model_type", "xgboost")
    mlflow.set_tag("experiment_phase", "week1_optimized")

    xgb_opt = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_opt.fit(X_train, y_train)

    y_pred = xgb_opt.predict(X_test)
    y_pred_proba = xgb_opt.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(xgb_opt, "model")

    print(f" F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

print("\n" + "=" * 70)
print("HUSSAIN'S EXPERIMENTS COMPLETED")
print("=" * 70)
print(" 4 experiments logged to MLflow")
print("Check DagsHub: https://dagshub.com/MadarwalaHussain/CustomerChurnPrediction/experiments")
print("=" * 70)
