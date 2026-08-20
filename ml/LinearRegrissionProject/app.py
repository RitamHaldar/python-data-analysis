import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="USA Housing Price Predictor",
    page_icon="🏡",
    layout="centered"
)

# App Title & Description
st.title("🏡 USA Housing Price Predictor")
st.markdown("Enter the demographic and property details below to estimate the house price.")

# Load the saved Ridge Regression model
@st.cache_resource
def load_model():
    with open("usa_housing_model_.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file `usa_housing_model_.pkl` not found. Ensure it is placed in the same directory.")
    st.stop()

# Input Form
st.subheader("Property & Area Metrics")

col1, col2 = st.columns(2)

with col1:
    avg_income = st.number_input(
        "Avg. Area Income ($)",
        min_value=1000.0,
        max_value=200000.0,
        value=68000.0,
        step=500.0,
        help="Average income of residents in the city."
    )
    avg_house_age = st.number_input(
        "Avg. Area House Age (Years)",
        min_value=0.0,
        max_value=50.0,
        value=6.0,
        step=0.1,
        help="Average age of houses in the same city."
    )
    avg_rooms = st.number_input(
        "Avg. Area Number of Rooms",
        min_value=1.0,
        max_value=20.0,
        value=7.0,
        step=0.1,
        help="Average number of rooms for houses in the same city."
    )

with col2:
    avg_bedrooms = st.number_input(
        "Avg. Area Number of Bedrooms",
        min_value=1.0,
        max_value=15.0,
        value=4.0,
        step=0.1,
        help="Average number of bedrooms for houses in the same city."
    )
    area_population = st.number_input(
        "Area Population",
        min_value=100.0,
        max_value=1000000.0,
        value=36000.0,
        step=500.0,
        help="Population of the city where the house is located."
    )

st.divider()

# Prediction Trigger
if st.button("Predict Price", type="primary", use_container_width=True):
    input_data = np.array([[avg_income, avg_house_age, avg_rooms, avg_bedrooms, area_population]])
    predicted_price = model.predict(input_data)[0]
    
    st.success(f"### Estimated Price: **${predicted_price:,.2f}**")