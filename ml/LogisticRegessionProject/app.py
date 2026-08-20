import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. PAGE CONFIGURATION & PREMIUM STYLING
# ==========================================
st.set_page_config(
    page_title="Personality Type Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
    }
    h1 {
        color: #2C3E50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
    }
    .subtitle {
        color: #7F8C8D;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    div.stButton > button:first-child {
        background-color: #6C5CE7;
        color: white;
        border-radius: 8px;
        height: 3rem;
        width: 100%;
        font-weight: bold;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #5849C4;
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3);
    }
    .result-card {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 2rem;
    }
    .result-text {
        font-size: 2.5rem;
        font-weight: 800;
        color: #6C5CE7;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD MODELS & SETUP
# ==========================================
@st.cache_resource
def load_assets():
    """Load the model and scaler from pickle files."""
    try:
        model = pickle.load(open("personality_model_.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models. Please ensure the .pkl files are in the same directory. ({e})")
        return None, None

model, scaler = load_assets()

# These are the exact 26 columns isolated from the ANOVA tests
FEATURES = [
    'party_liking', 'alone_time_preference', 'public_speaking_comfort',
    'talkativeness', 'excitement_seeking', 'social_energy', 'reading_habit',
    'group_comfort', 'leadership', 'adventurousness', 'deep_reflection',
    'risk_taking', 'sports_interest', 'decision_speed', 'routine_preference',
    'travel_desire', 'online_social_usage', 'work_style_collaborative',
    'spontaneity', 'gadget_usage', 'friendliness', 'listening_skill',
    'organization', 'planning', 'empathy', 'curiosity'
]

# Standard LabelEncoder sorts alphabetically: Ambivert (0), Extrovert (1), Introvert (2)
# (Adjust this mapping if your specific LabelEncoder mapped them differently)
PERSONALITY_MAP = {
    0: ("Ambivert ⚖️", "You have a beautiful balance of introverted and extroverted traits!"),
    1: ("Extrovert 🌟", "You thrive on social energy, excitement, and engaging with the world!"),
    2: ("Introvert 🧘", "You value deep reflection, alone time, and meaningful inner focus!")
}


# ==========================================
# 3. UI LAYOUT & INPUTS
# ==========================================
st.markdown("<h1>✨ Personality Type Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Answer the traits below to discover your true personality archetype.</p>", unsafe_allow_html=True)

# Using a form so the app doesn't re-run every time a slider is moved
with st.form("prediction_form"):
    st.markdown("### Rate yourself on the following traits (1 = Very Low, 10 = Very High)")
    st.write("---")
    
    # Create 3 columns for a clean, premium grid layout
    cols = st.columns(3)
    user_inputs = {}
    
    for i, feature in enumerate(FEATURES):
        col_index = i % 3
        # Format the feature name nicely (e.g., "party_liking" -> "Party Liking")
        clean_name = feature.replace('_', ' ').title()
        
        with cols[col_index]:
            user_inputs[feature] = st.slider(clean_name, min_value=1, max_value=10, value=5)
            
    st.write("---")
    submitted = st.form_submit_button("Predict My Personality")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if submitted:
    if model is not None and scaler is not None:
        with st.spinner("Analyzing your traits..."):
            # 1. Convert dictionary to dataframe (ensuring column order matches training data)
            input_df = pd.DataFrame([user_inputs])

            # Reuse the exact feature order recorded by the fitted scaler.
            expected_features = list(getattr(scaler, "feature_names_in_", FEATURES))
            if set(expected_features) != set(input_df.columns):
                st.error("The saved scaler features do not match the prediction form.")
                st.stop()
            input_df = input_df.loc[:, expected_features]
            
            # 2. Scale the input using your pre-fitted StandardScaler
            scaled_input = scaler.transform(input_df)
            
            # 3. Predict using the Logistic Regression model
            prediction_encoded = model.predict(scaled_input)[0]
            
            # 4. Map the result
            personality_type, description = PERSONALITY_MAP.get(prediction_encoded, ("Unknown", ""))
            
            # 5. Display Premium Output
            st.balloons()
            st.markdown(f"""
                <div class="result-card">
                    <h3>Your Personality Type is:</h3>
                    <p class="result-text">{personality_type}</p>
                    <p style="color: #7F8C8D; font-size: 1.1rem; margin-top: 10px;">{description}</p>
                </div>
            """, unsafe_allow_html=True)