import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame

# Create directories if not exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Save raw data
data.to_csv("data/raw/california_raw.csv", index=False)

# Split into train/test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train.drop("MedHouseVal", axis=1))
X_test = scaler.transform(test.drop("MedHouseVal", axis=1))

y_train = train["MedHouseVal"].values
y_test = test["MedHouseVal"].values

# Save processed data
pd.DataFrame(X_train, columns=housing.feature_names).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test, columns=housing.feature_names).to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_train, columns=["MedHouseVal"]).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["MedHouseVal"]).to_csv("data/processed/y_test.csv", index=False)

print("California housing data ingestion and preprocessing completed.")
