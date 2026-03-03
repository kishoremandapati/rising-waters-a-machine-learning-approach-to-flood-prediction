import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_excel("dataset/flood_data.xlsx")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

print("Columns in dataset:", data.columns)

# -------------------------------------------------
# Create Flood Label (Threshold Logic)
# -------------------------------------------------
data["flood"] = data["annual"].apply(lambda x: 1 if x > 3000 else 0)

# -------------------------------------------------
# Select Features
# -------------------------------------------------
features = ["annual", "jan-feb", "mar-may", "jun-sep", "oct-dec"]

X = data[features]
y = data["flood"]

# -------------------------------------------------
# Handle Missing Values
# -------------------------------------------------
X = X.fillna(X.mean())

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Train Random Forest Model
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Model Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# Feature Importance (Very Important for Viva)
# -------------------------------------------------
importances = model.feature_importances_
feature_names = X.columns

print("\nFeature Importance:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {round(importance, 4)}")

# Optional: Plot feature importance
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# -------------------------------------------------
# Save Model
# -------------------------------------------------
pickle.dump(model, open("models/flood_model.pkl", "wb"))

print("\n✅ Random Forest model trained and saved successfully!")