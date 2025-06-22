from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import aiohttp
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from typing import List
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_ENDPOINT = "https://models.github.ai/inference"
AZURE_MODEL = "openai/gpt-4.1"
AZURE_TOKEN = os.getenv("AZURE_API_KEY")  # Make sure this is set in your environment

azure_client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TOKEN),
)


app = FastAPI()
model = tf.keras.models.load_model("model/best_model.keras")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")


# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((128, 128)).convert("RGB")
    image_arr = np.array(image).astype(np.float32) / 255.0
    image_arr = np.expand_dims(image_arr, axis=0)

    dummy_tabular = np.zeros((1, 71), dtype=np.float32)

    prediction = model.predict([dummy_tabular, image_arr])
    prob = float(prediction[0][0])
    label = int(round(prob))
    diagnosis = "Malignant" if label == 1 else "Benign"

    return {
        "diagnosis": diagnosis,
        "confidence": prob
    }

# Define request model for chat
class ChatRequest(BaseModel):
    message: str

class Message(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str

class ChatHistoryRequest(BaseModel):
    chat_history: List[Message]

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
                    return {"answer": f"Error from OpenAI API: {resp.status}"}
                data = await resp.json()
                return {"answer": data['choices'][0]['message']['content'].strip()}
    except Exception as e:
        return {"answer": f"Request failed: {str(e)}"}
