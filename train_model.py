"""
train_model.py
--------------
Re-creates the exact Assignment 1 pipeline and saves the final model and
all preprocessing artifacts needed by the Flask app.

Pipeline (identical to Assignment 1 notebook):
  1. Load WA_Fn-UseC_-Telco-Customer-Churn (.xlsx or .csv)
  2. TotalCharges -> numeric, blanks (tenure = 0) filled with 0
  3. Drop duplicates and customerID
  4. Churn -> 0/1, new feature AvgMonthlySpend = TotalCharges / (tenure + 1)
  5. One-hot encoding with pd.get_dummies(drop_first=True)
  6. 80/20 stratified split (random_state = 42)
  7. StandardScaler on numeric columns (used by Logistic Regression only)
  8. Train Logistic Regression, Decision Tree, Random Forest
  9. Pick the best model by F1-score (Random Forest in Assignment 1)

Saved files:
  model.joblib          -> the best trained classifier
  model_columns.joblib  -> exact training feature order (31 columns)
  scaler.joblib         -> fitted StandardScaler (applied only if the best model needs scaling)
  model_info.json       -> model name, metrics, numeric columns, scaling flag

Run:  python train_model.py
"""

import os
import json
import warnings

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]


def load_data():
    xlsx_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
    csv_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Place the Telco dataset (.xlsx or .csv) next to train_model.py")


def main():
    # 1-3. Load and clean (same as Assignment 1)
    df = load_data()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop_duplicates().drop(columns=["customerID"])

    # 4-5. Feature engineering + one-hot encoding
    data = df.copy()
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data["AvgMonthlySpend"] = data["TotalCharges"] / (data["tenure"] + 1)
    data_encoded = pd.get_dummies(data, drop_first=True).astype(float)

    X = data_encoded.drop("Churn", axis=1)
    y = data_encoded["Churn"].astype(int)
    print("X shape:", X.shape, "| y shape:", y.shape)

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 7. Scaling (fitted on training data only)
    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test_s[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    # 8. Train the three Assignment 1 models
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced",
                                                   random_state=RANDOM_STATE), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                 random_state=RANDOM_STATE), False),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=8,
                                                 class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=-1), False),
    }

    results = {}
    for name, (model, needs_scaling) in models.items():
        Xtr = X_train_s if needs_scaling else X_train
        Xte = X_test_s if needs_scaling else X_test
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, pred), 3),
            "Precision": round(precision_score(y_test, pred), 3),
            "Recall": round(recall_score(y_test, pred), 3),
            "F1-Score": round(f1_score(y_test, pred), 3),
            "ROC-AUC": round(roc_auc_score(y_test, proba), 3),
        }

    print(pd.DataFrame(results).T)

    # 9. Best model by F1-score
    best_name = max(results, key=lambda n: results[n]["F1-Score"])
    best_model, best_scaled = models[best_name]
    print("Best model by F1-Score:", best_name)

    # Save model + preprocessing artifacts
    joblib.dump(best_model, os.path.join(BASE_DIR, "model.joblib"))
    joblib.dump(list(X.columns), os.path.join(BASE_DIR, "model_columns.joblib"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.joblib"))

    info = {
        "model_name": best_name,
        "needs_scaling": best_scaled,
        "num_cols": NUM_COLS,
        "n_features": X.shape[1],
        "metrics": results[best_name],
        "all_results": results,
    }
    with open(os.path.join(BASE_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("Saved: model.joblib, model_columns.joblib, scaler.joblib, model_info.json")


if __name__ == "__main__":
    main()
