from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from scipy.stats import boxcox
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---- Load model artifacts (must be at the top, before any function uses them) ----
model = joblib.load("xgb_model.pkl")
lambdas = joblib.load("boxcox_lambdas.pkl")
feature_columns = joblib.load("feature_columns.pkl")
continuous_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# ---- Pydantic model for the JSON API ----
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ---- Shared prediction logic ----
def predict_churn(data: dict) -> float:
    row = pd.DataFrame([data])
    for col in continuous_features:
        if col in lambdas and row[col].iloc[0] > 0:
            row[col] = boxcox(row[col], lmbda=lambdas[col])
    row = pd.get_dummies(row, dtype=int)
    row = row.reindex(columns=feature_columns, fill_value=0)
    return float(model.predict_proba(row)[0][1])

# ---- JSON API endpoint ----
@app.post("/predict")
def predict(customer: Customer):
    return {"churn_probability": predict_churn(customer.model_dump())}

# ---- Web form endpoints ----
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse(request, "form.html", {"result": None})

@app.post("/predict-form", response_class=HTMLResponse)
def form_post(request: Request,
              gender: str = Form(...), SeniorCitizen: int = Form(...),
              Partner: str = Form(...), Dependents: str = Form(...),
              tenure: int = Form(...), PhoneService: str = Form(...),
              MultipleLines: str = Form(...), InternetService: str = Form(...),
              OnlineSecurity: str = Form(...), OnlineBackup: str = Form(...),
              DeviceProtection: str = Form(...), TechSupport: str = Form(...),
              StreamingTV: str = Form(...), StreamingMovies: str = Form(...),
              Contract: str = Form(...), PaperlessBilling: str = Form(...),
              PaymentMethod: str = Form(...), MonthlyCharges: float = Form(...),
              TotalCharges: float = Form(...)):
    data = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    prob = predict_churn(data)
    return templates.TemplateResponse(request, "form.html", {"result": round(prob, 3)})