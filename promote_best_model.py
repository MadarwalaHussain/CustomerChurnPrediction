"""
Automatic Model Promotion Workflow
1. Find best model from latest training run
2. Register in Model Registry
3. Promote to Staging
4. (Optional) Auto-promote to Production if F1 > threshold
"""

from mlflow import MlflowClient
import dagshub
import mlflow
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# SSL fix
try:
    import fix_ssl
    fix_ssl.apply_ssl_fix()
    print("✓ SSL fix applied")
except:
    pass

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""


# Initialize DagsHub
dagshub.init(
    repo_owner='MadarwalaHussain',
    repo_name='CustomerChurnPrediction',
    mlflow=True
)

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
client = MlflowClient()

print("=" * 70)
print("MODEL PROMOTION WORKFLOW")
print("=" * 70)

# Step 1: Get the latest training run
print("\nStep 1: Finding latest model training run...")

try:
    experiment = mlflow.get_experiment_by_name("bank-churn-training")

    if not experiment:
        print(" Experiment 'bank-churn-training' not found!")
        print("   Run 'python main.py' first to train models.")
        sys.exit(1)

    # Get latest run with best_model tag
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.best_model != ''",
        order_by=["start_time DESC"],
        max_results=1
    )

    if len(runs) == 0:
        print(" No training runs found!")
        print("   Run 'python main.py' first to train models.")
        sys.exit(1)

    latest_run = runs.iloc[0]
    run_id = latest_run.run_id
    best_model = latest_run["params.best_model"]
    best_f1 = float(latest_run["metrics.best_test_f1"])

    print(f" Latest run found:")
    print(f"  Run ID: {run_id}")
    print(f"  Best Model: {best_model}")
    print(f"  F1 Score: {best_f1:.4f}")

except Exception as e:
    print(f"Error finding run: {e}")
    sys.exit(1)

# Step 2: Register model in Model Registry
print("\nStep 2: Registering model in Model Registry...")

model_name = "BankChurnProduction"
model_uri = f"runs:/{run_id}/model"

try:
    # Check if model already registered
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"  Model '{model_name}' already exists")
    except:
        print(f"  Creating new model '{model_name}'")

    # Register this version
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "model_type": best_model,
            "f1_score": str(best_f1),
            "source": "automated_training"
        }
    )

    version_number = model_version.version
    print(f"Model registered as version {version_number}")

except Exception as e:
    print(f" Error registering model: {e}")
    sys.exit(1)

# Step 3: Promote to Staging
print("\nStep 3: Promoting to Staging...")

try:
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"✓ Version {version_number} promoted to Staging")

    # Add description
    client.update_model_version(
        name=model_name,
        version=version_number,
        description=f"Model: {best_model} | F1: {best_f1:.4f} | Auto-promoted from training"
    )

except Exception as e:
    print(f" Error promoting to Staging: {e}")
    sys.exit(1)

# Step 4: Check if should auto-promote to Production
print("\nStep 4: Checking Production promotion criteria...")

AUTO_PROMOTE_THRESHOLD = 0.60  # F1 must be >= 0.60 for auto-promotion
MANUAL_REVIEW_REQUIRED = True   # Set to False to enable auto-promotion

if best_f1 >= AUTO_PROMOTE_THRESHOLD and not MANUAL_REVIEW_REQUIRED:
    print(f"  F1 ({best_f1:.4f}) >= threshold ({AUTO_PROMOTE_THRESHOLD})")
    print("  Auto-promoting to Production...")

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Production",
            archive_existing_versions=True  # Archive old Production models
        )
        print(f"Version {version_number} promoted to Production!")

    except Exception as e:
        print(f"Failed to promote to Production: {e}")
        print("  Model remains in Staging")
else:
    if best_f1 < AUTO_PROMOTE_THRESHOLD:
        print(f"  F1 ({best_f1:.4f}) < threshold ({AUTO_PROMOTE_THRESHOLD})")
    if MANUAL_REVIEW_REQUIRED:
        print("  Manual review required before Production")
    print("  Model will remain in Staging for testing")

print("\n" + "=" * 70)
print("PROMOTION WORKFLOW COMPLETE")
print("=" * 70)

# Summary
print("\nModel Status Summary:")
print(f"  Model Name: {model_name}")
print(f"  Version: {version_number}")
print(f"  Algorithm: {best_model}")
print(f"  F1 Score: {best_f1:.4f}")
print(f"  Current Stage: Staging")

print("\nNext Steps:")
if best_f1 >= AUTO_PROMOTE_THRESHOLD and MANUAL_REVIEW_REQUIRED:
    print("  1. Test model in Staging environment")
    print("  2. Validate on real data")
    print("  3. Run: python promote_to_production.py")
else:
    print("  1. Test model in Staging environment")
    print("  2. Improve model if F1 is too low")
    print("  3. Re-train and try again")

print("\nView on DagsHub:")
print(f"  https://dagshub.com/MadarwalaHussain/CustomerChurnPrediction/models")
print("=" * 70)
