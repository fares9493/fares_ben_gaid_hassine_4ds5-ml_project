from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict
import joblib
import numpy as np

# Load model and preprocessors
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
except FileNotFoundError:
    print("❌ Model files not found. Ensure the training pipeline has been executed.")

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn based on input features.",
    version="1.0.0",
)


# Input schema
class InputData(BaseModel):
    Account_Length: int = Field(..., example=1)
    Area_Code: int = Field(..., example=2)
    Customer_Service_Calls: int = Field(..., example=3)
    International_Plan: int = Field(..., example=4)
    Number_of_Voicemail_Messages: int = Field(..., example=5)
    Total_Day_Calls: int = Field(..., example=6)
    Total_Day_Charge: float = Field(..., example=7.0)
    Total_Day_Minutes: float = Field(..., example=8.0)
    Total_Night_Calls: int = Field(..., example=9)
    Total_Night_Charge: float = Field(..., example=10.0)
    Total_Night_Minutes: float = Field(..., example=11.0)
    Total_Evening_Calls: int = Field(..., example=12)
    Total_Evening_Charge: float = Field(..., example=13.0)
    Total_Evening_Minutes: float = Field(..., example=14.0)
    International_Calls: int = Field(..., example=15)
    Voicemail_Plan: int = Field(..., example=16)
    Extra_Feature_1: float = Field(..., example=17.0)
    Extra_Feature_2: float = Field(..., example=18.0)
    Extra_Feature_3: float = Field(..., example=19.0)


# Response schema
class PredictionResponse(BaseModel):
    prediction: int = Field(..., example=1)


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    try:
        # Convert input data to NumPy array
        input_features = np.array(list(data.dict().values())).reshape(1, -1)
        input_scaled = scaler.transform(input_features)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)[0]

        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}


# Custom validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
        },
    )
