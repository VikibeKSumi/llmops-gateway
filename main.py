from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# 1. The NEW Standard URL (OpenAI-compatible)
# This is the modern router address that works for all new models
API_URL = "https://router.huggingface.co/v1/chat/completions"

@app.get("/")
def home():
    return {"message": "LLM Gateway is Live (SmolLM2)"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise HTTPException(status_code=500, detail="API Token is missing!")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # 2. The Payload (Chat Format)
    # We use "HuggingFaceTB/SmolLM2-1.7B-Instruct" because it is:
    # - Free
    # - Ungated (No permission needed)
    # - Native to the new Router
    payload = {
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "messages": [
            {"role": "user", "content": request.prompt}
        ],
        "max_tokens": 50
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return {
                "error": "Provider Error", 
                "status": response.status_code, 
                "details": response.text
            }
        
        # 3. Parse the Chat Response
        data = response.json()
        # The answer is hidden deeper in the chat format
        answer = data['choices'][0]['message']['content']
        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}
