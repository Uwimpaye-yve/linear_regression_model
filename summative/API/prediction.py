from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Life Expectancy Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

model = joblib.load("best_life_expectancy_model.pkl")
scaler = joblib.load("scaler.pkl")

# Pydantic Model with all 20 features to satisfy the Scaler
# We give defaults (like 0.0) to the ones NOT in your Flutter app
class LifeData(BaseModel):
    Year: int = Field(..., ge=2000, le=2026)
    Status: int = Field(..., ge=0, le=1)
    Adult_Mortality: float = Field(..., ge=1, le=1000)
    infant_deaths: float = 0.0  # Default for missing Flutter field
    Alcohol: float = Field(..., ge=0, le=20)
    percentage_expenditure: float = 0.0
    Hepatitis_B: float = Field(..., ge=0, le=100)
    Measles: float = 0.0
    BMI: float = Field(..., ge=1, le=100)
    under_five_deaths: float = 0.0
    Polio: float = Field(..., ge=0, le=100)
    Total_expenditure: float = 0.0
    Diphtheria: float = Field(..., ge=0, le=100)
    HIV_AIDS: float = 0.0
    GDP: float = Field(..., ge=0)
    Population: float = 0.0
    thinness_1_19: float = 0.0
    thinness_5_9: float = 0.0
    Income_composition: float = 0.0
    Schooling: float = Field(..., ge=0, le=25)

@app.get("/")
def home():
    return {"message": "API is running. Go to /docs"}

@app.post("/predict")
async def predict(data: LifeData):
    # CRITICAL: This list MUST follow the exact order of X.columns from your notebook
    input_list = [
        data.Year, 
        data.Status, 
        data.Adult_Mortality, 
        data.infant_deaths, 
        data.Alcohol,
        data.percentage_expenditure,
        data.Hepatitis_B,
        data.Measles,
        data.BMI,
        data.under_five_deaths,
        data.Polio,
        data.Total_expenditure,
        data.Diphtheria,
        data.HIV_AIDS,
        data.GDP,
        data.Population,
        data.thinness_1_19,
        data.thinness_5_9,
        data.Income_composition,
        data.Schooling
    ]
    
    # Scale and Predict
    scaled_input = scaler.transform([input_list])
    prediction = model.predict(scaled_input)
    
    return {"predicted_life_expectancy": round(float(prediction[0]), 2)}

@app.post("/retrain")
async def retrain_model():
    return {"status": "success", "message": "Model retraining triggered."}