# from fastapi import FastAPI, HTTPException, Form
# import joblib
# import os
# import sys

# app = FastAPI(title="Support Ticket Auto-Triage API")

# model = None
# vectorizer = None

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models", "svm") 

# if not os.path.exists(MODEL_DIR):
#     MODEL_DIR = os.path.join(BASE_DIR, "models", "svm")

# try:
#     vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
#     model = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
#     print("Models loaded successfully!")
# except Exception as e:
#     print(f"Error loading models: {e}")

# @app.get("/health")
# def health_check():
#     return {"status": "ok", "model_loaded": model is not None}

# @app.post("/predict")
# def predict_ticket(
#     subject: str = Form(..., description="Subject"),
#     body: str = Form(..., description="Content")
# ):
#     if not model or not vectorizer:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     full_text = f"{subject} \n {body}"

#     vectorized_text = vectorizer.transform([full_text])

#     prediction = model.predict(vectorized_text)[0]
    
#     return {
#         "predicted_queue": prediction
#     }

import os
import torch
from fastapi import FastAPI, HTTPException, Form
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Support Ticket Auto-Triage API (DistilBERT)")

# Path to the saved transformer model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "transformer")

classifier = None

# Initialize the model on startup
@app.on_event("startup")
def load_model():
    global classifier
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model directory not found at {MODEL_PATH}")
            return

        print(f"Loading DistilBERT model from {MODEL_PATH}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Move to GPU if available (specifically for your RTX 5070 setup)
        device = 0 if torch.cuda.is_available() else -1
        
        # Initialize pipeline
        classifier = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )
        print("DistilBERT Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "model_loaded": classifier is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict")
def predict_ticket(
    subject: str = Form(..., description="The subject line of the ticket"),
    body: str = Form(..., description="The main content/body of the ticket")
):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    full_text = f"{subject}\n{body}"

    try:
        result = classifier(full_text, truncation=True, max_length=128)[0]

        return {
            "predicted_queue": result['label'],
            "confidence": f"{result['score']:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))