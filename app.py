from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

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

        # Make prediction and calculate probabilities
        probabilities = model.predict_proba(scaled_data)[0]
        prediction = model.predict(scaled_data)[0]
        confidence = round(probabilities[prediction] * 100, 2)

        # Determine if it's a risk case
        risk = prediction == 1
        result = "Risk of Heart Disease" if risk else "No Risk of Heart Disease"

        # For pie chart: survival chance = 100 - risk %
        survival = 100 - confidence if risk else confidence
        death_risk = 100 - survival

        # Debug
        print("Prediction:", prediction, "| Confidence:", confidence, "| Survival:", survival)

        return render_template("index.html", prediction=result, risk=risk, confidence=survival, death_risk=death_risk)

    except Exception as e:
        print("Prediction Error:", e)
        return render_template("index.html", prediction="An error occurred during prediction.", risk=False)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
