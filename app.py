from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained components
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

#Home Route (UI Page)
@app.route("/")
def home():
    return render_template("index.html", crops=None, reasons=None)

#Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Convert to array
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

        # Apply SAME scaling as training
        features_scaled = scaler.transform(features)

        probs = model.predict_proba(features_scaled)

        
        top3_idx = np.argsort(probs[0])[-3:][::-1]
        
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probs[0][top3_idx]

        reasons = generate_explanation(N, P, K, temp, humidity, ph, rainfall)
        
        return render_template(
            "index.html",
            crops=top3_crops,
            probs=top3_probs,
            reasons=reasons
        )
    
    except Exception as e:
        return render_template(
            "index.html",
            crops=None,
            probs=None,
            reasons=[str(e)]
        )

def generate_explanation(N, P, K, temp, humidity, ph, rainfall):
    reasons = []

    if rainfall > 150:
        reasons.append("high rainfall")
    elif rainfall < 50:
        reasons.append("low rainfall")

    if temp > 30:
        reasons.append("high temperature")
    elif temp < 20:
        reasons.append("moderate temperature")

    if humidity > 70:
        reasons.append("high humidity")

    if ph >= 6 and ph <= 7:
        reasons.append("optimal pH level")

    if N > 80:
        reasons.append("high nitrogen content")

    return reasons

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)