import pickle
import numpy as np

# Load saved files
model = pickle.load(open("model/crop_model.pkl", "rb"))
encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

def validate_input(n, p, k, temp, humidity, ph, rainfall):
    values = [n, p, k, temp, humidity, ph, rainfall]
    return all(v >= 0 for v in values)

def preprocess_input(n, p, k, temp, humidity, ph, rainfall):
    data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    return scaler.transform(data)

def predict_crop(n, p, k, temp, humidity, ph, rainfall):
    data = preprocess_input(n, p, k, temp, humidity, ph, rainfall)
    prediction = model.predict(data)
    crop = encoder.inverse_transform(prediction)
    return crop[0]