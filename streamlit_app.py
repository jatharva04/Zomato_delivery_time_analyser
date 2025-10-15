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

# --- STREAMLIT APP ---
st.title("Zomato Delivery Time Prediction 🚀")

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
    shap_values = explainer(transformed_data) # Use explainer call for Explanation object
    
    # Generate the SHAP plot object
    plot = shap.force_plot(
        explainer.expected_value, shap_values.values[0,:],
        features=transformed_data[0,:], feature_names=preprocessor.get_feature_names_out()
    )
    
    # Save the prediction and the plot to session state
    st.session_state.prediction = prediction[0]
    st.session_state.shap_plot = plot

# --- Main Page Tabs ---
tab1, tab2 = st.tabs(["Prediction & Explainability", "Model Performance"])

with tab1:
    st.header("Real-Time Prediction")
    if st.session_state.prediction is not None:
        st.metric(label="Predicted Delivery Time", value=f"{st.session_state.prediction:.2f} minutes")
        st.header("Why this prediction? (Live SHAP Analysis)")
        st.markdown("This chart shows how each factor pushed the prediction higher (red) or lower (blue) for this specific order.")
        # Render the saved SHAP plot
        st_shap(st.session_state.shap_plot, height=200)
    else:
        st.info("Enter details in the sidebar and click 'Predict Delivery Time' to see the results here.")

with tab2:
    st.header("Overall Model Performance")
    st.markdown("The model was evaluated using **Root Mean Squared Error (RMSE)**.")
    st.metric(label="Final Model RMSE", value="3.95 minutes")
    st.success("This level of accuracy means the model's predictions are typically off by less than 4 minutes.")