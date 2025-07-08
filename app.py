from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")      # Ensure model.pkl is in the same folder
scaler = joblib.load("scaler.pkl")    # Ensure scaler.pkl is in the same folder

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form and convert to float
        input_data = [float(x) for x in request.form.values()]

        # Scale the input
        scaled_data = scaler.transform([input_data])

        # Make prediction and calculate survival chance
        probabilities = model.predict_proba(scaled_data)[0]
        prediction = model.predict(scaled_data)[0]
        confidence = round(probabilities[prediction] * 100, 2)

        # Assuming: 1 = Risk, 0 = No Risk
        risk = prediction == 1
        result = "Risk of Heart Disease" if risk else "No Risk of Heart Disease"

        # Debug log
        print("Prediction value:", prediction, "Confidence:", confidence)

        return render_template("index.html", prediction=result, risk=risk, confidence=confidence)

    except Exception as e:
        print("Prediction Error:", e)
        return render_template("index.html", prediction="An error occurred during prediction.", risk=False)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
