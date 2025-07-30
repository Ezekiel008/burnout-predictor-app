import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# Load trained model
model = joblib.load('burnout_rf_model.pkl')

# Page config
st.set_page_config(page_title="Burnout Predictor", layout="wide")
st.title("🧠 Employee Burnout Prediction App")
st.markdown("Predict an employee's **Burn Rate** using workplace and mental health indicators.")

# Initialize session state for predictions
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar inputs
st.sidebar.header("🔧 Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
company_type = st.sidebar.selectbox("Company Type", ["Service", "Product"])
wfh_available = st.sidebar.selectbox("WFH Setup Available", ["Yes", "No"])
designation = st.sidebar.slider("Designation Level", 0.0, 1.0, 0.2, 0.1)
resource_allocation = st.sidebar.slider("Resource Allocation", 0.0, 1.0, 0.2, 0.1)
mental_fatigue_score = st.sidebar.slider("Mental Fatigue Score", 0.0, 1.0, 0.5, 0.05)
tenure_days = st.sidebar.number_input("Tenure in Days (Normalized)", 0.0, 1.0, 0.3, 0.01)

# Additional features (to match 10 features total)
age = st.sidebar.slider("Age (Normalized)", 0.0, 1.0, 0.4, 0.01)
burn_rate_prev = st.sidebar.slider("Previous Burn Rate (Optional)", 0.0, 1.0, 0.5, 0.01)
joining_year = st.sidebar.slider("Joining Year (Normalized)", 0.0, 1.0, 0.6, 0.01)

# Encode categorical features
gender_val = 1 if gender == "Female" else 0
company_val = 1 if company_type == "Product" else 0
wfh_val = 1 if wfh_available == "Yes" else 0

# Prediction
if st.sidebar.button("🔍 Predict Burnout"):
    input_data = np.array([[gender_val, company_val, wfh_val,
                            designation, resource_allocation, mental_fatigue_score,
                            tenure_days, age, burn_rate_prev, joining_year]])
    
    burn_rate = model.predict(input_data)[0]
    st.session_state.prediction_history.append(burn_rate)

    # Result display
    st.subheader("📊 Prediction Result")
    st.success(f"🎯 **Predicted Burn Rate: {burn_rate:.2f}**")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=burn_rate,
        title={'text': "Burnout Risk Level"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    st.markdown("### 🧠 What Does This Mean?")
    st.info("""
    - **Burn Rate** represents how likely the employee is experiencing burnout.
    - Values closer to `1.0` suggest **high burnout risk**, while those closer to `0.0` imply **low risk**.

    **Recommendations:**
    - High Burnout: Consider adjusting workload, improving WFH support, and mental wellness programs.
    - Medium Burnout: Monitor regularly, offer check-ins or flexible support.
    - Low Burnout: Maintain the current support system and employee satisfaction.
    """)

    # Save to CSV
    history_file = "burn_rate_predictions.csv"
    new_data = {
        "Gender": gender,
        "Company Type": company_type,
        "WFH Setup": wfh_available,
        "Designation": designation,
        "Resource Allocation": resource_allocation,
        "Mental Fatigue Score": mental_fatigue_score,
        "Tenure (Days)": tenure_days,
        "Age": age,
        "Previous Burn Rate": burn_rate_prev,
        "Joining Year": joining_year,
        "Predicted Burn Rate": burn_rate
    }

    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        history_df = pd.DataFrame([new_data])

    history_df.to_csv(history_file, index=False)
    st.success("📁 Prediction saved to **burn_rate_predictions.csv**")

# Show prediction history
if st.session_state.prediction_history:
    st.markdown("### 📈 Prediction History")
    history_df = pd.DataFrame({
        "Prediction #": range(1, len(st.session_state.prediction_history) + 1),
        "Burn Rate": st.session_state.prediction_history
    })
    st.dataframe(history_df)
    st.line_chart(history_df.set_index("Prediction #"))
