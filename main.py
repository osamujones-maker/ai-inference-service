import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from datetime import datetime, timezone

app = FastAPI(
    title="AI Cloud Inference Service",
    description="A serverless cloud microservice running a custom machine learning model.",
    version="1.0.0"
)

# Load the AI model once when the service starts (not on every request).
# DistilBERT fine-tuned on SST-2: a PyTorch transformer that classifies
# English text as POSITIVE or NEGATIVE sentiment.
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
classifier = pipeline("sentiment-analysis", model=MODEL_NAME, framework="pt")

# --- Storage layer: log every prediction to Firestore (NoSQL) ---
ENABLE_FIRESTORE = os.getenv("ENABLE_FIRESTORE", "false").lower() == "true"
db = None
if ENABLE_FIRESTORE:
    from google.cloud import firestore
    db = firestore.Client()  # On Cloud Run this uses the service's own identity

def save_prediction(text: str, label: str, confidence: float) -> None:
    """Store one prediction in the 'predictions' collection. Never breaks the API."""
    if db is None:
        return
    try:
        db.collection("predictions").add({
            "input_text": text,
            "label": label,
            "confidence": confidence,
            "model": MODEL_NAME,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        print(f"Firestore logging failed: {e}")

# Define the expected input schema using Pydantic
class InferenceRequest(BaseModel):
    text_input: str
    max_length: int = 50
 
# Define the response schema
class InferenceResponse(BaseModel):
    status: str
    input_text: str
    prediction: str
 
@app.get("/")
def health_check():
    """Health check endpoint to verify service is running."""
    return {"status": "online", "message": "AI Inference API is active."}
 
@app.post("/predict", response_model=InferenceResponse)
def run_inference(payload: InferenceRequest):
    """
    Core AI prediction endpoint. 
    Students: Replace this mock logic with your actual PyTorch/TensorFlow/HuggingFace model!
    """
    try:
        # --- YOUR AI MODEL INFERENCE CODE GOES HERE ---
        if not payload.text_input.strip():
            raise HTTPException(status_code=400, detail="text_input must not be empty.")

        # Run the PyTorch model. max_length limits how many tokens the model reads.
        token_limit = max(8, min(payload.max_length, 512))
        result = classifier(payload.text_input, truncation=True, max_length=token_limit)[0]

        label = result["label"]          # "POSITIVE" or "NEGATIVE"
        confidence = result["score"]     # probability between 0 and 1
        model_result = f"{label} (confidence: {confidence:.2%})"
        save_prediction(payload.text_input, label, confidence)
        # ----------------------------------------------

        return {
            "status": "success",
            "input_text": payload.text_input,
            "prediction": model_result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))