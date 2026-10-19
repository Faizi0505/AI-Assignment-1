# Customer Churn Prediction Web App (Flask) — AI Assignment 2

Flask web application that serves the **Random Forest** churn model from Assignment 1.
A user fills in a customer's details and gets a clear **Churn / No Churn** prediction,
the churn probability, and a risk level.

## Project structure

```
churn_app/
├── app.py                     # Flask backend: routes, validation, preprocessing, prediction
├── train_model.py             # Re-runs the Assignment 1 pipeline and saves the model files
├── model.joblib               # Saved Random Forest (best model by F1-score)
├── model_columns.joblib       # Exact 31 training columns, in training order
├── scaler.joblib              # Fitted StandardScaler (saved for completeness; RF does not need it)
├── model_info.json            # Model name, test metrics, numeric columns, scaling flag
├── requirements.txt           # Python packages
├── test_app.py                # Local tests (predictions + error handling)
├── pythonanywhere_wsgi.py     # WSGI file contents for PythonAnywhere
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (only needed to retrain)
├── templates/
│   └── index.html             # Input form + result page
└── static/
    └── style.css              # Styling
```

## Input-to-prediction flow

1. `GET /` renders `index.html` with a form for all 19 customer features.
2. The user submits → `POST /predict`.
3. `validate_form()` checks every field: required, allowed dropdown values, numeric ranges,
   whole-number tenure, and logical consistency (e.g. no internet ⇒ add-ons must be
   "No internet service"). Errors are shown on the page; the form keeps its values.
4. `preprocess()` applies the **same steps as Assignment 1**:
   * `AvgMonthlySpend = TotalCharges / (tenure + 1)`
   * one-hot encoding with `pd.get_dummies`
   * `reindex(columns=model_columns, fill_value=0)` → same 31 columns in the same order
   * scaling only if the saved model needs it (Random Forest does not)
5. `model.predict()` and `model.predict_proba()` → shown as "Churn"/"No Churn",
   probability %, and risk level (Low < 40% ≤ Medium < 70% ≤ High).

## Run locally

```bash
cd churn_app
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py                    # open http://127.0.0.1:5000
python test_app.py               # optional: 24 automated checks
```

If you see a scikit-learn version warning or a model loading error, rebuild the model
with your installed versions: `python train_model.py` (takes a few seconds).

## Deploy on PythonAnywhere (free account)

1. **Sign up / log in** at https://www.pythonanywhere.com (Beginner plan is enough).
   Your site will be `https://YOUR_USERNAME.pythonanywhere.com`.

2. **Upload the project.** Zip the `churn_app` folder on your computer.
   *Files* tab → upload `churn_app.zip` to `/home/YOUR_USERNAME/`.
   Then open a **Bash console** (*Consoles* tab → Bash) and run:
   ```bash
   cd ~
   unzip churn_app.zip
   ls churn_app            # should list app.py, model.joblib, templates, static ...
   ```
   (Alternative: push the folder to GitHub and `git clone` it in the Bash console.)

3. **Create a virtualenv and install packages** (in the same Bash console):
   ```bash
   mkvirtualenv --python=/usr/bin/python3.13 churn-venv
   pip install -r ~/churn_app/requirements.txt
   ```
   If `python3.13` is not available on your account, use the newest one listed by
   `ls /usr/bin/python3*` and install without pinned versions:
   `pip install flask pandas numpy scikit-learn joblib`, then run step 4.

4. **Make sure the model loads with the server's library versions:**
   ```bash
   cd ~/churn_app
   python -c "import app; print('Model loaded:', app.model_info['model_name'])"
   ```
   If this prints an error or an `InconsistentVersionWarning`, retrain on the server
   (same pipeline, same random_state):
   ```bash
   python train_model.py
   ```

5. **Create the web app.** *Web* tab → **Add a new web app** → Next →
   **Manual configuration** (not "Flask") → choose the **same Python version** as
   the virtualenv → Next.

6. **Configure it** on the Web tab:
   * **Source code:** `/home/YOUR_USERNAME/churn_app`
   * **Working directory:** `/home/YOUR_USERNAME/churn_app`
   * **Virtualenv:** `/home/YOUR_USERNAME/.virtualenvs/churn-venv`
   * **WSGI configuration file:** click the link, delete everything, paste the contents
     of `pythonanywhere_wsgi.py`, replace `YOUR_USERNAME`, and **Save**.
   * **Static files:** URL `/static/` → Directory `/home/YOUR_USERNAME/churn_app/static`

7. Click the green **Reload** button, then open `https://YOUR_USERNAME.pythonanywhere.com`.

8. **Test online** from your phone or another browser: high-risk example → *Churn*,
   low-risk example → *No Churn*, and an empty/invalid submission → error message.

### Troubleshooting

| Symptom | Fix |
|---|---|
| "Something went wrong" / error page | Web tab → **Error log** (last lines show the cause) |
| `ModuleNotFoundError: No module named 'app'` | Wrong path in WSGI file; must be `/home/YOUR_USERNAME/churn_app` |
| `ModuleNotFoundError: sklearn/pandas` | Virtualenv path not set on the Web tab, or packages installed outside it |
| `FileNotFoundError: model.joblib` | Model files not uploaded into `churn_app/` |
| Model unpickling error / version warning | `workon churn-venv && cd ~/churn_app && python train_model.py`, then Reload |
| Page has no styling | Add the `/static/` static-files mapping, then Reload |
| Disk quota exceeded during pip install | `rm -rf ~/.cache/pip`; or skip the virtualenv (leave Virtualenv blank) and use PythonAnywhere's preinstalled Flask/pandas/scikit-learn, then run `python3.13 train_model.py` |
| Site disabled after a few months | Free sites must be extended: Web tab → "Run until 3 months from today" |
