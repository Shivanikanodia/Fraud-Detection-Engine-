from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model pipeline
pipeline = joblib.load("fraud_xgb_pipeline.joblib")
EXPECTED_COLS = list(pipeline.feature_names_in_)

# -------------------------
# Home page (UI)
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# -------------------------
# Predict from FORM submit
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # ----------------------------------
        # Parse datetime field
        # ----------------------------------
        txn_dt = datetime.fromisoformat(form["txn_datetime"])

        # ----------------------------------
        # Build input row (MATCH TRAINING)
        # ----------------------------------
        data = {
    "merchant": form["merchant"],
    "category": form["category"],
    "amt": float(form["amt"]),
    "gender": form["gender"],
    "city": form["city"],
    "state": form["state"],
    "zip": int(form["zip"]),
    "job": "unknown",
    "dob": "1970-01-01",
    "unix_time": int(txn_dt.timestamp()),
    "lat": 0,
    "long": 0,
    "city_pop": 0,
    "merch_lat": 0,
    "merch_long": 0,
    "year": txn_dt.year,
    "month": txn_dt.month,
    "day": txn_dt.day,
    "hour": txn_dt.hour,
    "day_of_week": txn_dt.weekday(),
    "is_weekend": 1 if txn_dt.weekday() >= 5 else 0,
    "merchant_category_combo": f"{form['merchant']}_{form['category']}",
    "distance_km": 0
}

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure schema alignment
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = 0

        df = df[EXPECTED_COLS]

        # Predict probability
        prob = pipeline.predict_proba(df)[:, 1][0]
        risk_pct = round(prob * 100, 2)

        # -------------------------
        # Simple decision logic
        # -------------------------
        if prob >= 0.7:
            decision = "BLOCK"
            label = "Fraud"
        elif prob >= 0.3:
            decision = "MANUAL_REVIEW"
            label = "Suspicious"
        else:
            decision = "APPROVE"
            label = "Legit"

        result = {
            "prob": round(prob, 4),
            "risk_pct": risk_pct,
            "label": label,
            "decision": decision,
            "drivers": []  # placeholder (SHAP later)
        }

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result={"error": str(e)})




if __name__ == "__main__":
    app.run(debug=True, port=3000)