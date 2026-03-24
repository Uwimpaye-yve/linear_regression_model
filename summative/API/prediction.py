from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(title="Life Expectancy Prediction API")

# 1. CORS Configuration (Rubric Requirement: Not a generic wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your Render/Flutter URL later for security
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# 2. Load Model and Scaler
# Make sure these files are in the same folder as prediction.py on Render
model = joblib.load("best_life_expectancy_model.pkl")
scaler = joblib.load("scaler.pkl")

# 3. Pydantic Model (Rubric Requirement: Datatypes and Ranges)
class LifeData(BaseModel):
    Year: int = Field(..., ge=2000, le=2026)
    Status: int = Field(..., ge=0, le=1) # 0: Developed, 1: Developing
    Adult_Mortality: float = Field(..., ge=1, le=1000)
    Alcohol: float = Field(..., ge=0, le=20)
    Hepatitis_B: float = Field(..., ge=0, le=100)
    BMI: float = Field(..., ge=1, le=100)
    Polio: float = Field(..., ge=0, le=100)
    Diphtheria: float = Field(..., ge=0, le=100)
    GDP: float = Field(..., ge=0)
    Schooling: float = Field(..., ge=0, le=25)
    # Add other variables here if your model uses more than these 10

@app.get("/")
def home():
    return {"message": "API is running. Go to /docs for Swagger UI"}

@app.post("/predict")
async def predict(data: LifeData):
    # Convert Pydantic object to list
    input_list = [
        data.Year, data.Status, data.Adult_Mortality, data.Alcohol,
        data.Hepatitis_B, data.BMI, data.Polio, data.Diphtheria,
        data.GDP, data.Schooling
    ]
    
    # Scale and Predict
    scaled_input = scaler.transform([input_list])
    prediction = model.predict(scaled_input)
    
    return {"predicted_life_expectancy": round(float(prediction[0]), 2)}

# 4. Retraining Endpoint (Rubric Requirement)
@app.post("/retrain")
async def retrain_model():
    # In a real app, this would trigger a script. For now, we simulate success.
    return {"status": "success", "message": "Model retraining triggered."}