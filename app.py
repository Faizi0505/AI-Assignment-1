"""
app.py - Customer Churn Prediction Web Application (Flask)

Input-to-prediction flow:
  1. GET  /         -> show the customer input form (templates/index.html)
  2. POST /predict  -> read form values -> validate -> build a one-row DataFrame
                       -> apply the SAME preprocessing as Assignment 1
                          (AvgMonthlySpend, one-hot encoding, training column order,
                           scaling only if the saved model needs it)
                       -> model.predict / predict_proba -> show "Churn" / "No Churn"
"""

import os
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # absolute paths work on PythonAnywhere

app = Flask(__name__)

# ------------------------------------------------------------------
# Load the saved model and preprocessing artifacts once at start-up
# ------------------------------------------------------------------
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
model_columns = joblib.load(os.path.join(BASE_DIR, "model_columns.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
with open(os.path.join(BASE_DIR, "model_info.json")) as f:
    model_info = json.load(f)

NUM_COLS = model_info["num_cols"]          # tenure, MonthlyCharges, TotalCharges, AvgMonthlySpend
NEEDS_SCALING = model_info["needs_scaling"]  # False for Random Forest

# ------------------------------------------------------------------
# Allowed values for every categorical feature (same as the dataset)
# ------------------------------------------------------------------
YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["Yes", "No", "No internet service"]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}
INTERNET_ADDON_FIELDS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                         "TechSupport", "StreamingTV", "StreamingMovies"]

# Numeric fields: (label, min, max, integer?)
NUMERIC_RULES = {
    "tenure": ("Tenure (months)", 0, 100, True),
    "MonthlyCharges": ("Monthly charges", 0, 500, False),
    "TotalCharges": ("Total charges", 0, 50000, False),
}

LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner",
    "Dependents": "Dependents", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
def validate_form(form):
    """Return (customer_dict, errors). customer_dict uses training column names/types."""
    errors = []
    customer = {}

    # Categorical values must be one of the allowed options
    for field, options in CATEGORICAL_OPTIONS.items():
        value = (form.get(field) or "").strip()
        if not value:
            errors.append(f"Please select a value for {LABELS[field]}.")
        elif value not in options:
            errors.append(f"'{value}' is not a valid option for {LABELS[field]}.")
        else:
            customer[field] = value

    # Numeric values must be present, numeric, and within a sensible range
    for field, (label, lo, hi, is_int) in NUMERIC_RULES.items():
        raw = (form.get(field) or "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number (you entered '{raw}').")
            continue
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
            errors.append(f"{label} must be a real number.")
            continue
        if is_int and not value.is_integer():
            errors.append(f"{label} must be a whole number.")
            continue
        if not lo <= value <= hi:
            errors.append(f"{label} must be between {lo} and {hi}.")
            continue
        customer[field] = int(value) if is_int else value

    if errors:
        return customer, errors

    # Logical consistency checks (combinations that never occur in the dataset)
    if customer["PhoneService"] == "No" and customer["MultipleLines"] != "No phone service":
        errors.append("If Phone service is 'No', Multiple lines must be 'No phone service'.")
    if customer["PhoneService"] == "Yes" and customer["MultipleLines"] == "No phone service":
        errors.append("Multiple lines cannot be 'No phone service' when Phone service is 'Yes'.")
    for f in INTERNET_ADDON_FIELDS:
        if customer["InternetService"] == "No" and customer[f] != "No internet service":
            errors.append(f"If Internet service is 'No', {LABELS[f]} must be 'No internet service'.")
        if customer["InternetService"] != "No" and customer[f] == "No internet service":
            errors.append(f"{LABELS[f]} cannot be 'No internet service' when the customer has internet.")
    if customer["tenure"] == 0 and customer["TotalCharges"] > customer["MonthlyCharges"]:
        errors.append("A new customer (tenure 0) cannot have total charges above one month's charges.")

    customer["SeniorCitizen"] = int(customer["SeniorCitizen"])   # numeric in the dataset
    return customer, errors


# ------------------------------------------------------------------
# Preprocessing - identical to predict_churn() from Assignment 1
# ------------------------------------------------------------------
def preprocess(customer):
    row = pd.DataFrame([customer])
    row["AvgMonthlySpend"] = row["TotalCharges"] / (row["tenure"] + 1)   # engineered feature
    row = pd.get_dummies(row)                                            # one-hot encoding
    row = row.reindex(columns=model_columns, fill_value=0).astype(float)  # same 31 columns, same order
    if NEEDS_SCALING:
        row[NUM_COLS] = scaler.transform(row[NUM_COLS])
    return row


def risk_factors(c):
    """Plain-language reasons, based on the churn patterns found in Assignment 1 EDA."""
    reasons = []
    if c["Contract"] == "Month-to-month":
        reasons.append("Month-to-month contract (about 43% churn in the data)")
    if c["tenure"] <= 12:
        reasons.append("Short time with the company (12 months or less)")
    if c["InternetService"] == "Fiber optic":
        reasons.append("Fiber optic internet (about 42% churn in the data)")
    if c["PaymentMethod"] == "Electronic check":
        reasons.append("Pays by electronic check (about 45% churn in the data)")
    if c["TechSupport"] == "No":
        reasons.append("No tech support")
    if c["OnlineSecurity"] == "No":
        reasons.append("No online security")
    return reasons


def protective_factors(c):
    reasons = []
    if c["Contract"] == "Two year":
        reasons.append("Two-year contract (about 3% churn in the data)")
    elif c["Contract"] == "One year":
        reasons.append("One-year contract (about 11% churn in the data)")
    if c["tenure"] >= 48:
        reasons.append("Long-standing customer (4+ years)")
    if c["TechSupport"] == "Yes":
        reasons.append("Has tech support")
    if c["OnlineSecurity"] == "Yes":
        reasons.append("Has online security")
    if c["PaymentMethod"] in ("Bank transfer (automatic)", "Credit card (automatic)"):
        reasons.append("Automatic payment method")
    return reasons


def render_page(**kwargs):
    return render_template("index.html", options=CATEGORICAL_OPTIONS,
                           model_name=model_info["model_name"],
                           metrics=model_info["metrics"], **kwargs)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_page(form={}, result=None, errors=[])


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":          # someone opened /predict directly
        return render_page(form={}, result=None, errors=[])

    form = request.form.to_dict()
    customer, errors = validate_form(form)
    if errors:
        return render_page(form=form, result=None, errors=errors), 400

    try:
        X_new = preprocess(customer)
        pred = int(model.predict(X_new)[0])
        prob = float(model.predict_proba(X_new)[0][1])
    except Exception:  # never show a stack trace to the user
        app.logger.exception("Prediction failed")
        return render_page(form=form, result=None,
                           errors=["Something went wrong while making the prediction. Please try again."]), 500

    if prob >= 0.70:
        risk = "High"
    elif prob >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    result = {
        "label": "Churn" if pred == 1 else "No Churn",
        "is_churn": pred == 1,
        "probability": round(prob * 100, 1),
        "risk": risk,
        "risk_factors": risk_factors(customer),
        "protective_factors": protective_factors(customer),
    }
    return render_page(form=form, result=result, errors=[])


@app.errorhandler(404)
def not_found(_e):
    return render_page(form={}, result=None,
                       errors=["That page does not exist. Use the form below."]), 404


@app.errorhandler(405)
def bad_method(_e):
    return render_page(form={}, result=None,
                       errors=["That action is not allowed. Please use the form."]), 405


if __name__ == "__main__":
        app.run(debug=True, port=5001)
