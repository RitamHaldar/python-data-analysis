import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load the trained model and scaler ---
# Make sure these .pkl files are in the same directory as this app.py script
try:
    model = pickle.load(open("knnclassifiermodel.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Error: Model or Scaler file not found. Please ensure 'knnclassifiermodel.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# --- App UI Configuration ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Prediction App")
st.write("This application uses a K-Nearest Neighbors (KNN) Classifier to predict the likelihood of heart disease based on patient medical attributes.")

st.markdown("---")
st.subheader("Patient Health Data Input")

# --- Input Fields ---
# Organizing inputs into 3 columns for a cleaner UI
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200, step=1)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150, step=1)
    exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversable Defect")

st.markdown("---")

# --- Prediction Logic ---
if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    # 1. Gather all inputs into a Pandas DataFrame 
    # (Matches the format the scaler and model were likely trained on)
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    try:
        # 2. Scale the features using the loaded StandardScaler
        scaled_features = scaler.transform(input_data)
        
        # 3. Predict using the KNN classifier
        prediction = model.predict(scaled_features)
        
        # 4. Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("⚠️ The model predicts the **presence** of heart disease. Please consult a medical professional.")
        else:
            st.success("✅ The model predicts **no presence** of heart disease.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction. Ensure your input features exactly match the training dataset columns. Error details: {e}")