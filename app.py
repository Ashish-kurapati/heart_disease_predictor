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
        # Get user inputs from form and convert them to float
        data = [float(x) for x in request.form.values()]

        # Scale the input using the scaler
        data_scaled = scaler.transform([data])

        # Make prediction using the model
        prediction = model.predict(data_scaled)

        # Debug log (helpful on Render deployment)
        print("Prediction value:", prediction)

        # Assuming: 1 = Risk, 0 = No Risk
        risk = prediction[0] == 1
        result = "Risk of Heart Disease" if risk else "No Risk of Heart Disease"

        return render_template("index.html", prediction=result, risk=risk)

    except Exception as e:
        print("Error during prediction:", e)
        return render_template("index.html", prediction="An error occurred. Please check input values.", risk=False)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
