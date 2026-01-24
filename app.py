"""
Simplest Churn Prediction App - Direct Pickle Loading
"""

import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Churn Prediction", page_icon="🏦")

st.title("🏦 Bank Churn Prediction")

# Load model - BYPASS all custom functions
model_path = "final_models/model.pkl"

st.write(f"Looking for model at: {os.path.abspath(model_path)}")
st.write(f"File exists: {os.path.exists(model_path)}")

if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.stop()

# Try loading with standard pickle
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Pickle failed: {e}")

    # Try with different encoding
    try:
        with open(model_path, 'rb') as file:
            import pickle
            model = pickle.load(file, encoding='latin1')
        st.success("✅ Model loaded with latin1 encoding!")
    except Exception as e2:
        st.error(f"❌ Also failed: {e2}")

        # Show full error
        import traceback
        st.code(traceback.format_exc())
        st.stop()

st.write(f"Model type: {type(model).__name__}")

# Input fields
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance ($)", 0.0, 300000.0, 50000.0)
    num_products = st.slider("Number of Products", 1, 4, 2)

with col2:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    has_card = st.radio("Has Credit Card", ["Yes", "No"])
    is_active = st.radio("Is Active Member", ["Yes", "No"])
    salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 60000.0)

# Predict
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [1 if has_card == "Yes" else 0],
        'IsActiveMember': [1 if is_active == "Yes" else 0],
        'EstimatedSalary': [salary]
    })

    st.write("Input data:")
    st.dataframe(input_data)

    # Predict
    try:
        probability = model.predict_proba(input_data)[0][1]
        prediction = "Will Churn" if probability > 0.5 else "Will Stay"

        st.markdown("---")
        st.subheader("📊 Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Probability", f"{probability:.1%}")

        with col2:
            st.metric("Prediction", prediction)

        with col3:
            if probability >= 0.7:
                risk = "🔴 High"
            elif probability >= 0.5:
                risk = "🟡 Medium"
            else:
                risk = "🟢 Low"
            st.metric("Risk Level", risk)

        # Simple recommendation
        st.markdown("---")
        st.subheader("💡 Recommendation")

        if probability >= 0.7:
            st.error("**High Risk Customer**  \nImmediate retention action required. Assign to senior specialist.")
        elif probability >= 0.5:
            st.warning("**Medium Risk Customer**  \nProactive engagement needed. Add to monitoring list.")
        else:
            st.success("**Low Risk Customer**  \nMaintain standard service. Consider upsell opportunities.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())
