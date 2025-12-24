from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "LLM Gateway is Live"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        return {"error": "API Token is missing on server!"}

    try:
        # FIX: We provide the FULL URL to bypass the broken "lookup" logic
        # We are using Google's Flan-T5-Small (very fast and reliable)
        client = InferenceClient(
            model="https://api-inference.huggingface.co/models/google/flan-t5-small",
            token=api_token
        )
        
        # Flan-T5 is a "Text-to-Text" model, so we use text_generation
        response = client.text_generation(request.prompt, max_new_tokens=50)
        return {"response": response}

    except Exception as e:
        return {"error": str(e)}
