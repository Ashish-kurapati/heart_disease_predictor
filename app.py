from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        input_data = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        # Scale and predict
        data_scaled = scaler.transform([input_data])
        probabilities = model.predict_proba(data_scaled)[0]
        prediction = model.predict(data_scaled)[0]
        confidence = round(probabilities[prediction] * 100, 2)

        # Interpret result
        risk = prediction == 0
        result = "Risk of Heart Disease" if risk else "No Risk of Heart Disease"

        return render_template(
            "index.html",
            prediction=result,
            risk=risk,
            confidence=confidence
        )

    except Exception as e:
        print("Prediction Error:", e)
        return render_template("index.html", prediction="An error occurred.", risk=False)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
