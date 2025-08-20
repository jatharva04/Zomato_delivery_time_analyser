# validate_only.py

import pandas as pd
from great_expectations.dataset import PandasDataset
import json
from datetime import datetime

# =========================
# 1. Load the already cleaned dataset
# =========================
csv_path = "zomato_data.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. Validation (same as in exp2.py)
# =========================
ge_df = PandasDataset(df)

ge_df.expect_column_values_to_not_be_null("Delivery_person_ID")
ge_df.expect_column_values_to_be_between("Delivery_person_Age", 18, 60)
ge_df.expect_column_values_to_be_between("Delivery_person_Ratings", 1, 5)
ge_df.expect_column_values_to_be_between("distance (km)", 0, None)

results = ge_df.validate()

# =========================
# 3. Save validation results as JSON
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"validation_results_{timestamp}.json"

with open(output_file, "w") as f:
    json.dump(results.to_json_dict(), f, indent=2)   # FIXED here ✅

print(f"Validation results saved to {output_file}")
