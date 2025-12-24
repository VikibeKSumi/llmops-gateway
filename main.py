from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# We use the standard Chat API
API_URL = "https://router.huggingface.co/v1/chat/completions"

@app.get("/")
def home():
    return {"message": "LLM Gateway is Live (Qwen)"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise HTTPException(status_code=500, detail="API Token is missing!")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # We use Qwen because it is the current default "Free Tier" champion
    payload = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "messages": [
            {"role": "user", "content": request.prompt}
        ],
        "max_tokens": 100
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return {
                "error": "Provider Error", 
                "status": response.status_code, 
                "details": response.text
            }
        
        data = response.json()
        answer = data['choices'][0]['message']['content']
        return {"response": answer}

    except Exception as e:
        return {"error": str(e)}
