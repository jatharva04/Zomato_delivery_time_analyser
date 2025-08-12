# exp2.py

import pandas as pd
import janitor
from great_expectations.dataset import PandasDataset

# =========================
# 1. Read the dataset
# =========================
# Change path if needed
csv_path = "zomato_data.csv"  
df = pd.read_csv(csv_path)

# =========================
# 2. Remove empty columns & duplicates
# =========================
df = (
    df
    .remove_empty()
    .drop_duplicates()
)

# =========================
# 3. Handle missing values
# =========================
# Numeric
df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median())

df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].mean())

# Text
df['Time_Orderd'] = df['Time_Orderd'].fillna("Unknown")
df['Weather_conditions'] = df['Weather_conditions'].fillna("Unknown")
df['Road_traffic_density'] = df['Road_traffic_density'].fillna("Unknown")
df['City'] = df['City'].fillna("Unknown")

# Categorical/Other
df['multiple_deliveries'] = df['multiple_deliveries'].fillna(0).astype(int)
df['Festival'] = df['Festival'].fillna("No")

# =========================
# 4. Fix data types
# =========================
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], format='%H:%M', errors='coerce').dt.time

# =========================
# 5. Handle outliers
# =========================
# Age range
df = df[(df['Delivery_person_Age'] >= 18) & (df['Delivery_person_Age'] <= 60)]

# Ratings range
df = df[(df['Delivery_person_Ratings'] >= 1) & (df['Delivery_person_Ratings'] <= 5)]

# Distance IQR
q1 = df['distance (km)'].quantile(0.25)
q3 = df['distance (km)'].quantile(0.75)
iqr = q3 - q1
df = df[(df['distance (km)'] >= q1 - 1.5*iqr) & (df['distance (km)'] <= q3 + 1.5*iqr)]

# =========================
# 6. Validate schema (Great Expectations)
# =========================
ge_df = PandasDataset(df)

ge_df.expect_column_values_to_not_be_null("Delivery_person_ID")
ge_df.expect_column_values_to_be_between("Delivery_person_Age", 18, 60)
ge_df.expect_column_values_to_be_between("Delivery_person_Ratings", 1, 5)
ge_df.expect_column_values_to_be_between("distance (km)", 0, None)

results = ge_df.validate()
print("Validation Results:", results)

# =========================
# 7. Save cleaned dataset in place (for DVC)
# =========================
df.to_csv(csv_path, index=False)

print(f"✅ Cleaning complete. File saved: {csv_path}")
print("➡ Now run: dvc add", csv_path, "&& git commit -m 'Cleaned data' && dvc push")
