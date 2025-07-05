from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("heart_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    result = "Risk of Heart Disease" if prediction[0] == 1 else "No Risk of Heart Disease"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
   import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)

