from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# # Define local paths
# DATA_PATH = os.path.join("data", "Zomato_Data.csv") # Note: This file likely doesn't exist yet
# MODEL_PATH = "delivery_model.pkl"

# # Load dataset safely (Note: This is just for demonstration/setup; the model doesn't strictly need it)
# try:
#     if os.path.exists(DATA_PATH):
#         df = pd.read_csv(DATA_PATH)
#         print(f"✅ Dataset loaded successfully: {df.shape[0]} rows")
#     else:
#         # If the file doesn't exist, an empty DataFrame is used as a fallback
#         df = pd.DataFrame() 
#         print(f"⚠️ Dataset not found at {DATA_PATH}. Proceeding with fallback DataFrame.")
# except Exception as e:
#     print(f"⚠️ Error loading dataset: {e}")
#     df = pd.DataFrame() # fallback empty DataFrame

# class DeliveryInput(BaseModel):
#     """Defines the required input data for the prediction endpoint."""
#     distance: float = Field(..., gt=0, description="Distance in km")
#     weather: int = Field(..., ge=0, le=5, description="Weather severity (0-5, 0=clear, 5=stormy)")
#     traffic: int = Field(..., ge=0, le=5, description="Traffic level (0-5, 0=clear, 5=heavy)")

# def create_demo_model(path=MODEL_PATH):
#     """Creates and saves a simple demo LinearRegression model if none exists."""
#     # Features (X): Distance, Weather, Traffic
#     # Target (y): Delivery Time
#     np.random.seed(42)
#     X = np.random.rand(100, 3) * [5, 5, 5] # Scale features
#     # Simple linear relationship: Time = 5*Distance + 2*Weather + 3*Traffic + Noise
#     y = X[:, 0]*5 + X[:, 1]*2 + X[:, 2]*3 + np.random.normal(0, 1, 100)
    
#     model = LinearRegression().fit(X, y)
#     joblib.dump(model, path)
#     print(f"🧠 Demo model created and saved at {path}")
#     return model

# --- 1. DEFINE YOUR APPLICATION ---
app = FastAPI(title="Zomato Delivery Time Prediction", version="1.0" )

# --- 2. ADD YOUR CUSTOM TRANSFORMER ---
# This class definition is required for joblib to load your saved model.
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        time_ordered = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce')
        time_picked = pd.to_datetime(df['Time_Order_picked'], format='%H:%M', errors='coerce')
        prep_time = (time_picked - time_ordered).dt.total_seconds() / 60
        order_hour = time_ordered.dt.hour
        order_date = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
        day_of_week = order_date.dt.dayofweek
        
        extracted_features = pd.DataFrame({
            'preparation_time_mins': prep_time,
            'order_hour': order_hour,
            'day_of_week': day_of_week
        })
        return extracted_features.fillna(0)
    
    def get_feature_names_out(self, input_features=None):
        return ['preparation_time_mins', 'order_hour', 'day_of_week']

# --- 3. DEFINE THE CORRECT INPUT DATA MODEL ---
# This class accepts all 18 features your real model expects.
class Order(BaseModel):
    Delivery_person_Age: int
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weather_conditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: int
    Festival: str
    City: str
    distance_km: float # Using a JSON-friendly name

# --- 4. LOAD YOUR REAL TRAINED MODEL ---
import sys
sys.modules['__main__'].DatetimeFeatureExtractor = DatetimeFeatureExtractor

model_pipeline = joblib.load('models/delivery_model.pkl')
print("✅ Real Random Forest model loaded successfully!")

# --- 5. DEFINE THE API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "Zomato Delivery Time Prediction API is running."}

@app.post("/predict")
def predict_delivery_time(order: Order):
    """
    Receives all 18 order details and returns a delivery time prediction.
    """
    # Convert the input to a pandas DataFrame
    input_df = pd.DataFrame([order.model_dump()])
    
    # Rename the distance column to match what the model was trained on
    input_df.rename(columns={'distance_km': 'distance (km)'}, inplace=True)
    
    # The pipeline handles all preprocessing and prediction
    prediction = model_pipeline.predict(input_df)
    
    # Return the prediction
    return {"predicted_delivery_time_minutes": round(prediction[0], 2)}

# @app.get("/")
# def home():
#     """Returns a simple greeting message."""
#     return {"message": "Welcome to the Zomato Delivery Time Prediction API. Visit /docs for documentation."}

# @app.post("/predict")
# def predict(data: DeliveryInput):
#     """Predicts the delivery time based on input parameters."""
#     try:
#         # Convert input Pydantic model to a dictionary, then to a DataFrame
#         # Use .model_dump() for modern Pydantic versions
#         input_dict = data.model_dump()
#         df_input = pd.DataFrame([input_dict])
        
#         # Ensure the column order is correct for the model
#         feature_columns = ['distance', 'weather', 'traffic']
        
#         # Make prediction
#         prediction = model.predict(df_input[feature_columns])[0]
        
#         # Round and return the result
#         return {
#             "predicted_delivery_time_minutes": round(float(prediction), 2),
#             "units": "minutes"
#         }
#     except Exception as e:
#         # Log the error and return a detailed message
#         print(f"Prediction failed: {e}")
#         return {"error": f"Prediction failed: {e}"}
