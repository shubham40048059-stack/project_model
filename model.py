import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Load data
file_path = "house_price.csv"
df = pd.read_csv(file_path)

# Display dataset overview
print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Define features and target
features = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

X = df[features]
y = df["House_Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# Model creation and training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nSaved trained model to house_price_model.pkl")

# Sample prediction example
sample = X.iloc[[0]]
predicted_price = model.predict(sample)[0]
print("\nSample prediction for the first property:")
print(f"Predicted House Price: ${predicted_price:,.2f}")
