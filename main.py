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
        raise HTTPException(status_code=500, detail="API Token is missing!")

    try:
        # MAGIC FIX: We explicitly set provider="hf-inference"
        # This forces the library to use the new router automatically.
        client = InferenceClient(
            model="gpt2", 
            token=api_token,
            provider="hf-inference" 
        )
        
        # GPT-2 is a text completion model
        response = client.text_generation(request.prompt, max_new_tokens=50)
        return {"response": response}

    except Exception as e:
        # If this fails, it prints the exact reason
        return {"error": str(e)}
