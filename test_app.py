"""
test_app.py - local tests for the Flask churn app.
Checks that web predictions match the model directly, and that bad input is handled.
Run:  python test_app.py
"""
import re
import pandas as pd
import app as churn_app

client = churn_app.app.test_client()

HIGH = {"gender": "Female", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No", "tenure": "2",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": "80.0", "TotalCharges": "160.0"}
LOW = {"gender": "Male", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "Yes", "tenure": "60",
       "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
       "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
       "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
       "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
       "MonthlyCharges": "65.0", "TotalCharges": "3900.0"}


def web_predict(data):
    r = client.post("/predict", data=data)
    html = r.get_data(as_text=True)
    label = re.search(r"<h2>(Churn|No Churn)</h2>", html)
    prob = re.search(r"<strong>([\d.]+)%</strong>", html)
    return r.status_code, (label.group(1) if label else None), (float(prob.group(1)) if prob else None), html


def notebook_predict(customer):
    """predict_churn() exactly as written in Assignment 1, section 7."""
    row = pd.DataFrame([customer])
    row["AvgMonthlySpend"] = row["TotalCharges"] / (row["tenure"] + 1)
    row = pd.get_dummies(row)
    row = row.reindex(columns=churn_app.model_columns, fill_value=0).astype(float)
    pred = int(churn_app.model.predict(row)[0])
    prob = float(churn_app.model.predict_proba(row)[0][1])
    return ("Churn" if pred == 1 else "No Churn"), round(prob * 100, 1)


passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond: passed += 1; print(f"PASS  {name} {detail}")
    else:    failed += 1; print(f"FAIL  {name} {detail}")


# 1. Home page loads
r = client.get("/")
check("GET / returns 200 with form", r.status_code == 200 and b"Predict churn" in r.data)

# 2. Expected churn and expected no-churn cases (Assignment 1 examples)
s, label, prob, _ = web_predict(HIGH)
check("High-risk customer -> Churn", s == 200 and label == "Churn", f"({prob}%)")
s, label, prob, _ = web_predict(LOW)
check("Low-risk customer -> No Churn", s == 200 and label == "No Churn", f"({prob}%)")

# 3. Web predictions match the model on 10 real customers from the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
sample = df.sample(10, random_state=7)
for _, rec in sample.iterrows():
    cust = rec.drop(["customerID", "Churn"]).to_dict()
    form = {k: str(v) for k, v in cust.items()}
    s, label, prob, _ = web_predict(form)
    exp_label, exp_prob = notebook_predict(cust)
    check(f"Customer {rec.customerID} web == model", s == 200 and label == exp_label and prob == exp_prob,
          f"(web {label} {prob}% | model {exp_label} {exp_prob}% | actual {rec.Churn})")

# 4. Validation / error handling
s, label, _, html = web_predict({})
check("Empty form -> friendly error, no crash", s == 400 and label is None and "is required" in html)
bad = dict(HIGH, tenure="abc")
s, _, _, html = web_predict(bad)
check("Non-numeric tenure -> error", s == 400 and "must be a number" in html)
bad = dict(HIGH, tenure="-5")
s, _, _, html = web_predict(bad)
check("Negative tenure -> error", s == 400 and "between" in html)
bad = dict(HIGH, tenure="2.5")
s, _, _, html = web_predict(bad)
check("Fractional tenure -> error", s == 400 and "whole number" in html)
bad = dict(HIGH, Contract="Lifetime")
s, _, _, html = web_predict(bad)
check("Unknown dropdown value -> error", s == 400 and "not a valid option" in html)
bad = dict(HIGH, InternetService="No")
s, _, _, html = web_predict(bad)
check("Inconsistent internet add-ons -> error", s == 400 and "No internet service" in html)
bad = dict(HIGH, MonthlyCharges="nan")
s, _, _, html = web_predict(bad)
check("NaN monthly charges -> error", s == 400 and "real number" in html)
bad = dict(HIGH); bad.pop("PaymentMethod")
s, _, _, html = web_predict(bad)
check("Missing field -> error", s == 400 and "Payment method" in html)
check("Values kept after error", 'value="80.0"' in html)

# 5. Other routes
check("GET /predict shows form", client.get("/predict").status_code == 200)
check("Unknown page -> friendly 404", client.get("/nope").status_code == 404)

print(f"\n{passed} passed, {failed} failed")
