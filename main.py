from fastapi import FastAPI, HTTPException, Form
import joblib
import os
import sys

app = FastAPI(title="Support Ticket Auto-Triage API")

model = None
vectorizer = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "svm") 

if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(BASE_DIR, "models", "svm")

try:
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_ticket(
    subject: str = Form(..., description="Subject"),
    body: str = Form(..., description="Content")
):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    full_text = f"{subject} \n {body}"

    vectorized_text = vectorizer.transform([full_text])

    prediction = model.predict(vectorized_text)[0]
    
    return {
        "predicted_queue": prediction
    }