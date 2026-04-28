import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("engine_failure_dataset.csv")

# -----------------------------
# Clean column names (VERY IMPORTANT)
# -----------------------------
df.columns = df.columns.str.strip()

# -----------------------------
# Rename columns FIRST
# -----------------------------
# Clean column names (remove spaces, units, symbols)
df.columns = df.columns.str.strip()

df.rename(columns={
    'Temperature (°C)': 'Temperature',
    'Power_Output (kW)': 'Power_Output'
}, inplace=True)
# DEBUG: check columns
print("Columns:\n", df.columns)

# -----------------------------
# Feature Engineering
# -----------------------------

# Total vibration
df["Vibration_Total"] = np.sqrt(
    df["Vibration_X"]**2 +
    df["Vibration_Y"]**2 +
    df["Vibration_Z"]**2
)

# Thermal stress
df["Thermal_Stress"] = df["Temperature"] * df["Power_Output"]

# Efficiency proxy
df["Efficiency"] = df["Power_Output"] / df["Power_Output"].max()

# -----------------------------
# Features and Target
# -----------------------------
features = [
    "Temperature",
    "RPM",
    "Torque",
    "Power_Output",
    "Vibration_Total",
    "Thermal_Stress",
    "Efficiency"
]

X = df[features]
y = df["Fault_Condition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")