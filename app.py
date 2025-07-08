from flask import Flask, render_template, request
import joblib
import os

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
        scaled_data = scaler.transform([input_data])

        # Make prediction and calculate survival chance
        probabilities = model.predict_proba(scaled_data)[0]
        prediction = model.predict(scaled_data)[0]
        confidence = round(probabilities[prediction] * 100, 2)

        # Determine risk
        risk = prediction == 1
        result = "Risk of Heart Disease" if risk else "No Risk of Heart Disease"

        # Chart data
        chart_data = [confidence, 100 - confidence]
        chart_labels = ["Risk" if risk else "No Risk", "No Risk" if risk else "Risk"]
        chart_colors = ["#e74c3c", "#2ecc71"] if risk else ["#2ecc71", "#e74c3c"]

        # Render result
        return render_template(
            "index.html",
            prediction=result,
            risk=risk,
            confidence=confidence,
            chart_data=chart_data,
            chart_labels=chart_labels,
            chart_colors=chart_colors
        )

    except Exception as e:
        print("Prediction Error:", e)
        return render_template("index.html", prediction="An error occurred during prediction.", risk=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

