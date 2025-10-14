from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# Define local paths
DATA_PATH = os.path.join("data", "Zomato_Data.csv") # Note: This file likely doesn't exist yet
MODEL_PATH = "delivery_model.pkl"

# Load dataset safely (Note: This is just for demonstration/setup; the model doesn't strictly need it)
try:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows")
    else:
        # If the file doesn't exist, an empty DataFrame is used as a fallback
        df = pd.DataFrame() 
        print(f"⚠️ Dataset not found at {DATA_PATH}. Proceeding with fallback DataFrame.")
except Exception as e:
    print(f"⚠️ Error loading dataset: {e}")
    df = pd.DataFrame() # fallback empty DataFrame

class DeliveryInput(BaseModel):
    """Defines the required input data for the prediction endpoint."""
    distance: float = Field(..., gt=0, description="Distance in km")
    weather: int = Field(..., ge=0, le=5, description="Weather severity (0-5, 0=clear, 5=stormy)")
    traffic: int = Field(..., ge=0, le=5, description="Traffic level (0-5, 0=clear, 5=heavy)")

def create_demo_model(path=MODEL_PATH):
    """Creates and saves a simple demo LinearRegression model if none exists."""
    # Features (X): Distance, Weather, Traffic
    # Target (y): Delivery Time
    np.random.seed(42)
    X = np.random.rand(100, 3) * [5, 5, 5] # Scale features
    # Simple linear relationship: Time = 5*Distance + 2*Weather + 3*Traffic + Noise
    y = X[:, 0]*5 + X[:, 1]*2 + X[:, 2]*3 + np.random.normal(0, 1, 100)
    
    model = LinearRegression().fit(X, y)
    joblib.dump(model, path)
    print(f"🧠 Demo model created and saved at {path}")
    return model

# Load or create model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"📦 Model loaded from {MODEL_PATH}")
else:
    model = create_demo_model()

app = FastAPI(
    title="Zomato Delivery Time Prediction API",
    description="A simple API to predict delivery time based on distance, weather, and traffic.",
    version="1.0.0"
)

@app.get("/")
def home():
    """Returns a simple greeting message."""
    return {"message": "Welcome to the Zomato Delivery Time Prediction API. Visit /docs for documentation."}

@app.post("/predict")
def predict(data: DeliveryInput):
    """Predicts the delivery time based on input parameters."""
    try:
        # Convert input Pydantic model to a dictionary, then to a DataFrame
        # Use .model_dump() for modern Pydantic versions
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])
        
        # Ensure the column order is correct for the model
        feature_columns = ['distance', 'weather', 'traffic']
        
        # Make prediction
        prediction = model.predict(df_input[feature_columns])[0]
        
        # Round and return the result
        return {
            "predicted_delivery_time_minutes": round(float(prediction), 2),
            "units": "minutes"
        }
    except Exception as e:
        # Log the error and return a detailed message
        print(f"Prediction failed: {e}")
        return {"error": f"Prediction failed: {e}"}
