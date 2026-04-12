from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Financial Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the expected input JSON structure
class Transaction(BaseModel):
    step: int
    type: str
    amount: float


# Global variable to hold the model and mapping
model_data = None


@app.on_event("startup")
def load_model():
    """Load the model on startup."""
    global model_data
    model_path = "model/fraud_model.pkl"
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print("Error: Model file not found. Please run train.py first.")


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running. Use /predict for inference."}


@app.post("/predict")
def predict(transaction: Transaction):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    # 1. Extract data from request
    try:
        # Convert 'type' using the saved mapping
        type_val = model_data["type_map"].get(transaction.type.upper())
        if type_val is None:
            raise ValueError(f"Invalid transaction type: {transaction.type}")

        # 2. Prepare features for prediction
        # Features must be in the same order as training: [step, type, amount]
        features = np.array([[transaction.step, type_val, transaction.amount]])

        # 3. Make prediction
        prediction = model_data["model"].predict(features)[0]
        probability = model_data["model"].predict_proba(features)[0]

        # 4. Format response
        result = "Fraud" if prediction == 1 else "Not Fraud"
        confidence = float(max(probability))

        return {
            "prediction": result,
            "confidence": round(confidence, 4),
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
