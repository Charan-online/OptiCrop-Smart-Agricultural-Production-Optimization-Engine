from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and label encoder
model = pickle.load(open("model/crop_model.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction page
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Prepare input for the model
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Make prediction
        prediction = model.predict(features)

        # Decode prediction if label encoder exists
        crop = label_encoder.inverse_transform(prediction)[0]

        return render_template("result.html", prediction=crop)

    except Exception as e:
        return f"Error: {e}"

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
model = pickle.load(open("model/crop_model.pkl", "rb"))