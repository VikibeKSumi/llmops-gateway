from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# We use the NEW Router URL structure
# We use Flan-T5 because it is small, fast, and usually works on the free tier.
API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-small"

@app.get("/")
def home():
    return {"message": "LLM Gateway is Live (Router Version)"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise HTTPException(status_code=500, detail="API Token is missing!")

    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": request.prompt}

    try:
        # Direct request to the new Router
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # If the new Router fails (404/500), we print why
        if response.status_code != 200:
            return {
                "error": "Provider Error", 
                "status": response.status_code, 
                "details": response.text
            }
        
        # Success!
        return response.json()

    except Exception as e:
        return {"error": str(e)}
