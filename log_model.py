import mlflow
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract features from datetime columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Ensure the input is a DataFrame
        df = X.copy()

        # --- Feature 1: Food Preparation Time ---
        # Convert time columns to datetime objects, coercing errors
        time_ordered = pd.to_datetime(df['Time_Orderd'], errors='coerce')
        time_picked = pd.to_datetime(df['Time_Order_picked'], errors='coerce')
        # Calculate the difference in minutes
        prep_time = (time_picked - time_ordered).dt.total_seconds() / 60

        # --- Feature 2: Order Hour ---
        order_hour = time_ordered.dt.hour

        # --- Feature 3: Day of the Week ---
        order_date = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
        day_of_week = order_date.dt.dayofweek # Monday=0, Sunday=6

        # Create a new DataFrame with the extracted features
        extracted_features = pd.DataFrame({
            'preparation_time_mins': prep_time,
            'order_hour': order_hour,
            'day_of_week': day_of_week
        })

        # Handle any NaNs that might have resulted from conversion errors
        return extracted_features.fillna(0)

# Set your experiment name
mlflow.set_experiment("Zomato Delivery Time Prediction")

# --- Manually enter the results you got from your Colab notebook ---
best_params = {'regressor__max_depth': 20, 'regressor__min_samples_split': 10, 'regressor__n_estimators': 300}
test_rmse = 3.9503

# --- Load the model you downloaded from Colab ---
model_to_log = joblib.load('models/delivery_model.pkl')

# --- Start a run to log the pre-trained model ---
with mlflow.start_run(run_name="Tuned RF (trained on Colab)"):
    # Log the parameters and metrics you recorded from Colab
    mlflow.log_params(best_params)
    mlflow.log_metric("Test_RMSE", test_rmse)
    
    # Now, log the actual model artifact to your LOCAL MLflow
    mlflow.sklearn.log_model(model_to_log, "tuned_rf_from_colab")

print("Successfully logged the model trained on Colab to your local MLflow.")