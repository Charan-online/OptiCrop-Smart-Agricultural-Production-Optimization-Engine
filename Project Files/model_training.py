import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("dataset/Crop_recommendation.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==========================
# Prepare Features & Target
# ==========================

X = df.drop("label", axis=1)
y = df["label"]

# ==========================
# Encode Labels
# ==========================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==========================
# Train Model
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================
# Test Model
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Model Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\n============================")
print("MODEL EVALUATION")
print("============================")

print(f"Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Test Sample Prediction
# ==========================

sample = X_test.iloc[[0]]

prediction = model.predict(sample)

crop = label_encoder.inverse_transform(prediction)

print("\nSample Prediction")
print("Predicted Crop:", crop[0])

# ==========================
# Save Model
# ==========================

os.makedirs("model", exist_ok=True)

with open("model/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel saved successfully!")
print("Model Path : model/crop_model.pkl")

print("Label Encoder saved successfully!")
print("Encoder Path : model/label_encoder.pkl")