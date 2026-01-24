"""
Inspect what's in the model pickle file
"""

import pickle
import os

model_path = "final_models/model.pkl"

print("=" * 70)
print("INSPECTING MODEL FILE")
print("=" * 70)

print(f"\nFile: {os.path.abspath(model_path)}")
print(f"Exists: {os.path.exists(model_path)}")
print(f"Size: {os.path.getsize(model_path)} bytes")

print("\n" + "=" * 70)
print("ATTEMPTING TO LOAD...")
print("=" * 70)

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("\n✅ SUCCESS!")
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")

    # Check if it has predict_proba
    if hasattr(model, 'predict_proba'):
        print("✅ Has predict_proba method")

    # Try a dummy prediction
    import pandas as pd
    dummy = pd.DataFrame({
        'CreditScore': [650],
        'Geography': ['France'],
        'Gender': ['Male'],
        'Age': [35],
        'Tenure': [5],
        'Balance': [50000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [60000]
    })

    result = model.predict_proba(dummy)
    print(f"\n✅ Test prediction works!")
    print(f"Churn probability: {result[0][1]:.2%}")

except Exception as e:
    print(f"\n❌ FAILED: {e}")

    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

    print("\n" + "=" * 70)
    print("TRYING ALTERNATIVE METHODS...")
    print("=" * 70)

    # Try joblib
    try:
        import joblib
        model = joblib.load(model_path)
        print("✅ Joblib worked!")
    except Exception as e2:
        print(f"❌ Joblib failed: {e2}")

    # Try dill
    try:
        import dill
        with open(model_path, 'rb') as f:
            model = dill.load(f)
        print("✅ Dill worked!")
    except Exception as e3:
        print(f"❌ Dill failed: {e3}")
