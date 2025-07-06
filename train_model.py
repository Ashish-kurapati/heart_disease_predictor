import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load your dataset
df = pd.read_csv("heart.csv")  # Make sure heart.csv is in the same folder

# Fix the target column: 1 = has disease, 0 = no disease
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Optional: Print evaluation
y_pred = model.predict(X_scaled)
print(classification_report(y, y_pred))

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully.")
