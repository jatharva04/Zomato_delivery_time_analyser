import streamlit as st
import pandas as pd
import joblib
import shap
import streamlit.components.v1 as components
from sklearn.base import BaseEstimator, TransformerMixin
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="Zomato Delivery Time Predictor",
    page_icon="🚀",
    layout="wide"
)

# --- Custom Transformer Class ---
# This class MUST be defined so joblib can load the model pipeline
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        df = X.copy()
        time_ordered = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce')
        time_picked = pd.to_datetime(df['Time_Order_picked'], format='%H:%M', errors='coerce')
        prep_time = (time_picked - time_ordered).dt.total_seconds() / 60
        order_hour = time_ordered.dt.hour
        order_date = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
        day_of_week = order_date.dt.dayofweek
        extracted_features = pd.DataFrame({'preparation_time_mins': prep_time, 'order_hour': order_hour, 'day_of_week': day_of_week})
        return extracted_features.fillna(0)
    def get_feature_names_out(self, input_features=None):
        return ['preparation_time_mins', 'order_hour', 'day_of_week']

sys.modules['__main__'].DatetimeFeatureExtractor = DatetimeFeatureExtractor

# Helper function to render SHAP plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model_pipeline = joblib.load('models/delivery_model.pkl')
        return model_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_pipeline = load_model()

# --- Initialize Session State ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'shap_plot' not in st.session_state:
    st.session_state.shap_plot = None

# --- App Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Problem Statement", "Prediction App", "Ethical Considerations"])

# =========================================
# === PROBLEM STATEMENT PAGE ================
# =========================================
if page == "Problem Statement":
    st.title("🚀 Zomato Delivery Time Prediction")
    st.header("The Problem")
    st.markdown("""
    In the competitive world of food delivery, providing an accurate estimated time of arrival (ETA) is crucial for customer satisfaction.
    An inaccurate ETA can lead to frustrated customers and operational inefficiencies. This project aims to solve this challenge by building a
    machine learning model that can predict the delivery time of a Zomato order with high accuracy.
    """)

    st.header("The Goal")
    st.markdown("""
    The primary objective is to develop a tool that takes various factors into account—such as the delivery person's age and ratings,
    weather conditions, road traffic, and distance—to provide a reliable delivery time prediction. This dashboard serves as the final
    portfolio piece, allowing users to interact with the model, get real-time predictions, and understand the factors that influence the delivery time.
    """)

# =========================================
# === PREDICTION APP PAGE =================
# =========================================
elif page == "Prediction App":
    st.title("🔮 Prediction App")
    st.markdown("Enter the order details in the sidebar to get a delivery time prediction.")

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("Enter Order Details:")
        age = st.number_input("Delivery Person Age", min_value=18, max_value=50, value=35)
        ratings = st.slider("Delivery Person Rating", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
        distance = st.number_input("Distance (km)", min_value=0.1, max_value=20.0, value=3.5, step=0.1)
        traffic = st.selectbox("Road Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
        weather = st.selectbox("Weather Conditions", ["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"])
        festival = st.selectbox("Is it a festival?", ['No', 'Yes'])
        city = st.selectbox("City Type", ['Metropolitian', 'Urban', 'Semi-Urban'])
        predict_button = st.button("Predict Delivery Time", type="primary")

    # --- Logic for the predict button ---
    if predict_button and model_pipeline:
        input_data = {
            'Delivery_person_Age': age, 'Delivery_person_Ratings': ratings,
            'Restaurant_latitude': 19.0760, 'Restaurant_longitude': 72.8777,
            'Delivery_location_latitude': 19.0820, 'Delivery_location_longitude': 72.8860,
            'Order_Date': '15-10-2025', 'Time_Orderd': '19:15', 'Time_Order_picked': '19:30',
            'Weather_conditions': weather, 'Road_traffic_density': traffic,
            'Vehicle_condition': 1, 'Type_of_order': 'Meal', 'Type_of_vehicle': 'motorcycle',
            'multiple_deliveries': 1, 'Festival': festival, 'City': city,
            'distance (km)': distance
        }
        input_df = pd.DataFrame([input_data])

        prediction = model_pipeline.predict(input_df)
        preprocessor = model_pipeline.named_steps['preprocessor']
        model = model_pipeline.named_steps['regressor']
        transformed_data = preprocessor.transform(input_df)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(transformed_data)
        
        plot = shap.force_plot(
            explainer.expected_value, shap_values.values[0,:],
            features=transformed_data[0,:], feature_names=preprocessor.get_feature_names_out()
        )
        
        st.session_state.prediction = prediction[0]
        st.session_state.shap_plot = plot

    # --- Main Page Display ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Real-Time Prediction")
        if st.session_state.prediction is not None:
            st.metric(label="Predicted Delivery Time", value=f"{st.session_state.prediction:.2f} minutes")
        else:
            st.info("Enter details and click 'Predict' to see the results.")

    with col2:
        st.header("Model Performance")
        st.markdown("The model was evaluated using **Root Mean Squared Error (RMSE)**.")
        st.metric(label="Final Model RMSE", value="3.95 minutes")
        st.success("This accuracy means the model's predictions are typically off by less than 4 minutes.")

    st.markdown("---")
    st.header("Why this prediction? (Live SHAP Analysis)")

    if st.session_state.shap_plot is not None:
        st.markdown("This chart explains how each factor influenced this specific prediction, starting from a baseline prediction and showing what pushed the time higher (red) or lower (blue).")
        st_shap(st.session_state.shap_plot, height=200)

        with st.expander("How to read this chart 🤔"):
            st.markdown("""
            This is a SHAP (SHapley Additive exPlanations) Force Plot, which breaks down a single prediction.
            * **Base Value:** This is the average prediction the model would make if it had no information.
            * **Red Bars (Higher 🔺):** These are the features that pushed the prediction value **higher** than the base value. The longer the bar, the stronger the effect.
            * **Blue Bars (Lower 🔻):** These are the features that pushed the prediction value **lower** than the base value.
            * **Final Prediction:** The final number shown on the chart is the model's output for this specific set of inputs.
            """)
    else:
        st.info("Click 'Predict' to see the model's explanation here.")

# =========================================
# === ETHICAL CONSIDERATIONS PAGE =========
# =========================================
elif page == "Ethical Considerations":
    st.title("Ethical Considerations & Responsible AI")
    st.markdown("""
    This document outlines the principles of responsible AI applied to this project to ensure it is fair, transparent, and accountable.
    """)

    st.header("1. Fairness")
    st.markdown("""
    **Objective:** Ensure the model does not produce systematically biased predictions against certain groups.
    -   **Bias Check:** The dataset was analyzed for imbalances. Potential sources of bias could include `City` type or `Delivery_person_Age`.
    -   **Mitigation:** The model's performance was evaluated across different demographic segments (e.g., city types) to ensure it performs equitably. While no specific debiasing techniques have been implemented in this version, this analysis is a crucial first step for a production system.
    """)

    st.header("2. Privacy")
    st.markdown("""
    **Objective:** Protect user and delivery partner data.
    -   **Data Handling:** The application processes input data for prediction but does not store any personal identifiable information (PII).
    """)

    st.header("3. Transparency")
    st.markdown("""
    **Objective:** Make the model's decision-making process understandable.
    -   **Model Choice:** The model is a **Random Forest Regressor**, which is a well-understood ensemble model.
    -   **Explainability:** The dashboard integrates **SHAP (SHapley Additive exPlanations)**. This allows users to see a breakdown of which features contributed most to each individual prediction, making the model's logic transparent.
    """)

    st.header("4. Accountability")
    st.markdown("""
    **Objective:** Maintain a clear and auditable trail of the project's development.
    -   **Version Control:** The entire project codebase is versioned using **Git** and hosted on GitHub.
    -   **Experiment Tracking:** **MLflow** was used to log model parameters and metrics, ensuring reproducibility.
    -   **CI/CD Pipeline:** A **GitHub Actions** workflow automates testing and building, ensuring changes are validated.
    """)

    st.header("5. Consent")
    st.markdown("""
    **Objective:** Ensure users understand and agree to how their data is being used.
    -   **Explicit Use:** The dashboard's interface is clear. Users voluntarily provide input for the explicit purpose of receiving a delivery time prediction.
    -   **Data Collection:** In a real-world deployment, users would be informed via a privacy policy that their anonymized data is used to train and improve the prediction model.
    """)