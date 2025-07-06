from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("model.pkl")      # Make sure model.pkl exists
scaler = joblib.load("scaler.pkl")    # Make sure scaler.pkl exists

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form and convert to float
        data = [float(x) for x in request.form.values()]

        # Scale the input
        data_scaled = scaler.transform([data])

        # Predict using the model
        prediction = model.predict(data_scaled)

        # Debug log for Render
        print("Prediction value:", prediction)

        # Prepare result
        result = "No Risk of Heart Disease" if prediction[0] == 1 else "Risk of Heart Disease"

        return render_template("index.html", prediction=result)
    
    except Exception as e:
        print("Error during prediction:", e)
        return render_template("index.html", prediction="An error occurred. Please check input values.")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
