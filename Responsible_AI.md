# Responsible AI Report for Zomato Delivery Time Prediction

This document outlines the principles of responsible AI applied to this project.

## 1. Fairness
**Objective:** Ensure the model does not produce systematically biased predictions against certain groups.

-   **Bias Check:** The dataset was analyzed for imbalances. Potential sources of bias could include `City` type or `Delivery_person_Age`.
-   **Mitigation:** The model's performance should be evaluated across different demographic segments (e.g., city types) to ensure it performs equitably. We have not implemented specific debiasing techniques, but this would be a crucial next step for a production system.

## 2. Privacy
**Objective:** Protect user and delivery partner data.

-   **Data Handling:** The API processes input data for prediction but does not store any personal identifiable information (PII).
-   **Anonymization:** Location data (`latitude`, `longitude`) is used for distance calculation but should be anonymized or aggregated in a real-world scenario to protect user privacy.

## 3. Transparency
**Objective:** Make the model's decision-making process understandable.

-   **Model Choice:** The model is a **Random Forest Regressor**, which is a well-understood ensemble model.
-   **Explainability:** The Streamlit dashboard integrates **SHAP (SHapley Additive exPlanations)**. This allows users to see a breakdown of which features contributed most to each individual prediction, making the model's logic transparent.

## 4. Accountability
**Objective:** Maintain a clear and auditable trail of the project's development.

-   **Version Control:** The entire project codebase, including data schemas and models, is versioned using **Git** and hosted on GitHub.
-   **Experiment Tracking:** **MLflow** was used to log model parameters, metrics, and artifacts (`mlruns/`), ensuring reproducibility.
-   **CI/CD Pipeline:** The **GitHub Actions** workflow (`.github/workflows/`) automates testing and building, ensuring that any changes are validated before deployment.

## 5. Consent

**Objective:** Ensure that users understand and agree to how their data is being used by the application.

-   **Explicit Use**: The dashboard's user interface is clear and straightforward. Users voluntarily provide input data for the explicit purpose of receiving a delivery time prediction, implying consent for that specific transaction.

-   **Data Collection:** In a real-world deployment, users (both customers and delivery partners) would be informed via a privacy policy that their anonymized data is used to train and improve the prediction model.