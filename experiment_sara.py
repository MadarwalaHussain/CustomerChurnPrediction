"""
Sara's Experiments - Ensemble Methods
Week 1: Advanced ensemble techniques
"""

from bank_churns.utils.main_utils.utils import load_numpy_array
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
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
        print(" DagsHub initialized")
    except Exception as e:
        print(f"DagsHub init failed: {e}")

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
print("SARA'S EXPERIMENTS - Ensemble Methods")
print("=" * 70)
print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# ============================================
# EXPERIMENT 1: Gradient Boosting - Baseline
# ============================================
print("\n[1/3] Gradient Boosting - Baseline...")
with mlflow.start_run(run_name="sara_gb_baseline"):
    # Tags
    mlflow.set_tag("engineer", "sara")
    mlflow.set_tag("model_type", "gradient_boosting")
    mlflow.set_tag("experiment_phase", "week1_baseline")

    # Train
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = gb_model.predict(X_test)
    y_pred_proba = gb_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log params
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 3)

    # Log metrics
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(gb_model, "model")

    print(f" F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================
# EXPERIMENT 2: Gradient Boosting - Optimized
# ============================================
print("\n[2/3] Gradient Boosting - Optimized...")
with mlflow.start_run(run_name="sara_gb_optimized"):
    mlflow.set_tag("engineer", "sara")
    mlflow.set_tag("model_type", "gradient_boosting")
    mlflow.set_tag("experiment_phase", "week1_optimized")

    gb_opt = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    gb_opt.fit(X_train, y_train)

    y_pred = gb_opt.predict(X_test)
    y_pred_proba = gb_opt.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("subsample", 0.8)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(gb_opt, "model")

    print(f"✓ F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================
# EXPERIMENT 3: Voting Ensemble (RF + XGB + LR)
# ============================================
print("\n[3/3] Voting Ensemble - RF + XGBoost + LogisticRegression...")
with mlflow.start_run(run_name="sara_voting_ensemble"):
    mlflow.set_tag("engineer", "sara")
    mlflow.set_tag("model_type", "voting_ensemble")
    mlflow.set_tag("experiment_phase", "week1_advanced")

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Create base estimators
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    # Voting ensemble
    voting_model = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)

    y_pred = voting_model.predict(X_test)
    y_pred_proba = voting_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    mlflow.log_param("voting_type", "soft")
    mlflow.log_param("base_estimators", "RF+XGB+LR")
    mlflow.log_param("rf_n_estimators", 100)
    mlflow.log_param("xgb_n_estimators", 100)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(voting_model, "model")

    print(f"✓ F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

print("\n" + "=" * 70)
print("SARA'S EXPERIMENTS COMPLETED")
print("=" * 70)
print("3 experiments logged to MLflow")
print(" Check DagsHub: https://dagshub.com/MadarwalaHussain/CustomerChurnPrediction/experiments")
print("=" * 70)
