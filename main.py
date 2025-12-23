from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Support Ticket Auto-Triage API")

class Ticket(BaseModel):
    subject: str
    body: str

model = None
vectorizer = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "svm")

try:
    print(f"Loading models from: {MODEL_DIR}")
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
    print("Models loaded")
except Exception as e:
    print(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_ticket(ticket: Ticket):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    full_text = (ticket.subject or "") + "\n" + (ticket.body or "")
    
    vectorized_text = vectorizer.transform([full_text])

    prediction = model.predict(vectorized_text)[0]
    
    return {
        "predicted_queue": prediction,
    }