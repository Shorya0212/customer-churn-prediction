import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📉",
    layout="wide"
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown(
    """
    <h1 style='text-align: center;'>Customer Churn Prediction</h1>
    <p style='text-align: center; color: gray;'>
    ML-based churn risk assessment (probability driven)
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 10, 100, 30)
tenure = st.sidebar.slider("Tenure (months)", 0, 130, 10)
monthlycharge = st.sidebar.slider("Monthly Charges", 30, 150, 50)
gender = st.sidebar.radio("Gender", ["Male", "Female"])

st.sidebar.divider()

risk_threshold = st.sidebar.slider(
    "Churn Risk Sensitivity",
    min_value=0.5,
    max_value=0.9,
    value=0.8,
    step=0.05
)

predict_button = st.sidebar.button("Predict Churn")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Input Summary")
    st.write(f"Age: {age}")
    st.write(f"Tenure: {tenure} months")
    st.write(f"Monthly Charges: ${monthlycharge}")
    st.write(f"Gender: {gender}")
    st.write(f"Risk Threshold: {risk_threshold:.2f}")

with col2:
    st.subheader("Feature Snapshot")
    fig, ax = plt.subplots()
    ax.bar(
        ["Age", "Tenure", "Monthly Charge"],
        [age, tenure, monthlycharge],
        color="#4C72B0"   # neutral blue
    )
    ax.set_ylabel("Value")
    ax.set_title("Customer Profile")
    st.pyplot(fig)

st.divider()

if predict_button:
    gender_encoded = 1 if gender == "Female" else 0
    X_input = np.array([[age, gender_encoded, tenure, monthlycharge]])
    X_scaled = scaler.transform(X_input)

    probability = model.predict_proba(X_scaled)[0][1]

    st.subheader("Prediction Result")
    st.progress(int(probability * 100))

    if probability >= risk_threshold:
        st.error(f"High Risk of Churn ({probability:.2%})")
    elif probability >= 0.6:
        st.warning(f"Medium Risk of Churn ({probability:.2%})")
    else:
        st.success(f"Low Risk of Churn ({probability:.2%})")

    fig2, ax2 = plt.subplots()
    ax2.bar(
        ["Higher Churn", "Lower Churn"],
        [1 - probability, probability],
        color=["#E74C3C", "#2ECC71"]  # green, red
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")
    ax2.set_title("Churn Probability Distribution")
    st.pyplot(fig2)

else:
    st.info("Enter customer details and click Predict Churn")
