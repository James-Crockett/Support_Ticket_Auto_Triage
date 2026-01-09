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

from pydantic import BaseModel

class TicketIn(BaseModel):
    subject: str = ""
    body: str = ""

@app.post("/predict")
def predict_ticket(ticket: TicketIn):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    full_text = f"{ticket.subject}\n{ticket.body}".strip()

    result = classifier(full_text, truncation=True, max_length=128)[0]
    return {
        "predicted_queue": result["label"],
        "confidence": float(result["score"]),
    }
