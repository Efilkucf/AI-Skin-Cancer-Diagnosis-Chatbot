import os
import gc
# Set environment variables BEFORE importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import aiohttp
import gdown
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from typing import List
# Removed StaticFiles and FileResponse imports - not needed for API-only
from dotenv import load_dotenv
from pathlib import Path

# Configure TensorFlow for memory optimization
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Load environment variables
load_dotenv()
AZURE_ENDPOINT = "https://models.github.ai/inference"
AZURE_MODEL = "openai/gpt-4.1"
AZURE_TOKEN = os.getenv("AZURE_API_KEY")

# Initialize Azure client
azure_client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TOKEN),
)

# FastAPI app setup
app = FastAPI()

# Global model variable (singleton pattern)
_model = None

def download_model():
    """Download model if it doesn't exist"""
    model_path = Path("model/best_model.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print("Downloading model...")
        try:
            gdown.download(id="1zN6KmX_vY9R5XqE8oGjwa-9X8TYN4JFL", output=str(model_path), quiet=False)
            print("Model downloaded.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print("Model already exists, skipping download.")

def load_model_optimized():
    """Load model with memory optimization"""
    global _model
    if _model is None:
        print("Loading model into memory...")
        
        # Clear any existing TensorFlow session
        tf.keras.backend.clear_session()
        gc.collect()
        
        try:
            # Load model without compilation to save memory
            _model = tf.keras.models.load_model(
                "model/best_model.keras",
                compile=False  # Skip compilation for inference-only
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Force garbage collection
        gc.collect()
    
    return _model

def get_model():
    """Get the loaded model instance"""
    return load_model_optimized()

# Download model at startup
download_model()


# CORS middleware - Updated for separate frontend deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend URL later
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_optimized()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global _model
    if _model is not None:
        del _model
        _model = None
    tf.keras.backend.clear_session()
    gc.collect()

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Get model instance
        model = get_model()
        
        # Process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).resize((128, 128)).convert("RGB")
        image_arr = np.array(image).astype(np.float32) / 255.0
        image_arr = np.expand_dims(image_arr, axis=0)
        
        # Create dummy tabular data
        dummy_tabular = np.zeros((1, 71), dtype=np.float32)
        
        # Make prediction
        prediction = model.predict([dummy_tabular, image_arr], verbose=0)
        prob = float(prediction[0][0])
        label = int(round(prob))
        diagnosis = "Malignant" if label == 1 else "Benign"
        
        return {
            "diagnosis": diagnosis,
            "confidence": prob
        }
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
    
    finally:
        # Clean up variables
        locals().clear()
        gc.collect()

# Request models for chat
class Message(BaseModel):
    role: str
    content: str

class ChatHistoryRequest(BaseModel):
    chat_history: List[Message]

# Chat endpoint
@app.post("/ask")
async def ask_chat(req: ChatHistoryRequest):
    messages = [{"role": "system", "content": "You are an empathetic AI assistant helping users interpret skin cancer analysis results and provide medical guidance."}]
    messages += [msg.dict() for msg in req.chat_history]
    
    payload = {
        "model": AZURE_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 1
    }
    
    headers = {
        "Authorization": f"Bearer {AZURE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{AZURE_ENDPOINT}/chat/completions", json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {"answer": f"Error from API: {resp.status} - {error_text}"}
                
                data = await resp.json()
                return {"answer": data['choices'][0]['message']['content'].strip()}
    
    except Exception as e:
        return {"answer": f"Request failed: {str(e)}"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": _model is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
