import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ✅ Load Trained Model & Scaler
model_data = joblib.load("injury_prediction_model.pkl")
xgb_model = model_data['model']
scaler = model_data['scaler']

# ✅ Define the Input Form
st.title("Athlete Injury Prediction System 🏃‍♂️")
st.write("Enter athlete details below to predict injury risk:")

# User Inputs
age = st.number_input("Age", min_value=15, max_value=50, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=140, max_value=220, value=175)
weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70)
position = st.selectbox("Position", ["Defender", "Midfielder", "Forward"])
training_hours = st.number_input("Training Hours Per Week", min_value=0.0, max_value=30.0, value=10.0)
training_intensity = st.selectbox("Training Intensity", ["Low", "Medium", "High"])
match_count = st.number_input("Matches Per Week", min_value=0, max_value=10, value=2)
rest_days = st.number_input("Rest Between Events (Days)", min_value=0, max_value=10, value=3)
recovery_days = st.number_input("Recovery Days Per Week", min_value=0, max_value=7, value=2)
fatigue_score = st.slider("Fatigue Score", 0.0, 10.0, 5.0)
performance_score = st.slider("Performance Score", 0.0, 100.0, 85.0)
team_contribution = st.slider("Team Contribution Score", 0, 100, 75)
load_balance = st.slider("Load Balance Score", 0, 100, 80)
acl_risk = st.slider("ACL Risk Score", 0.0, 10.0, 3.5)

# ✅ Convert Inputs to Match Training Format
data = pd.DataFrame({
    "Age": [age],
    "Gender": [1 if gender == "Male" else 0],  # Encode Gender
    "Height_cm": [height],
    "Weight_kg": [weight],
    "Position": [0 if position == "Defender" else (1 if position == "Midfielder" else 2)],  # Encode Position
    "Training_Intensity": [0 if training_intensity == "Low" else (1 if training_intensity == "Medium" else 2)],  # Encode Training Intensity
    "Training_Hours_Per_Week": [training_hours],
    "Recovery_Days_Per_Week": [recovery_days],
    "Match_Count_Per_Week": [match_count],
    "Rest_Between_Events_Days": [rest_days],
    "Fatigue_Score": [fatigue_score],
    "Performance_Score": [performance_score],
    "Team_Contribution_Score": [team_contribution],
    "Load_Balance_Score": [load_balance],
    "ACL_Risk_Score": [acl_risk]
})

# ✅ Scale Data Using Trained Scaler
data_scaled = scaler.transform(data)

# ✅ Make Prediction
y_prob = xgb_model.predict_proba(data_scaled)[:, 1][0]  # Probability of injury
best_threshold = 0.68
y_pred = int(y_prob >= best_threshold)  # Convert probability to 0 or 1

# ✅ Display Result
if st.button("Predict Injury Risk"):
    st.write(f"### 🏥 Prediction: {'High Injury Risk (1)' if y_pred == 1 else 'Low Injury Risk (0)'}")
    st.write(f"**Injury Probability: {y_prob:.2f}**")

st.write("Developed by [Your Name] 🚀")
