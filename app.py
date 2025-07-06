from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")  # 👈 make sure this file exists in your folder

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input from form and convert to float
    data = [float(x) for x in request.form.values()]
    
    # Scale the input
    data_scaled = scaler.transform([data])  # 👈 this is important

    # Predict using the model
    prediction = model.predict(data_scaled)

    # Prepare result
    result = "Risk of Heart Disease" if prediction[0] == 1 else "No Risk of Heart Disease"
    
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
