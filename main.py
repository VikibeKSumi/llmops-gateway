from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from langsmith import traceable 

app = FastAPI()

# 1. Define the Model once
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "LLM Gateway is Live (Qwen)"}

# 2. The "Worker" Function (This is what we Trace)
# We move ALL the logic here so LangSmith sees the inputs and outputs.
@traceable 
def call_huggingface_api(user_prompt: str):
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("API Token is missing!")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # 1. NEW URL (Points to the 7B Model)
    url = "https://router.huggingface.co/models/Qwen/Qwen2.5-Coder-7B-Instruct/v1/chat/completions"
    
    # 2. NEW PAYLOAD (Matches the 7B Model)
    payload = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 500
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Provider Error {response.status_code}: {response.text}")
        
    return response.json()

# 3. The Endpoint (The "Manager")
# It just receives the request and delegates to the worker.
@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        # CRITICAL: We call the decorated function here!
        # This triggers the LangSmith trace.
        api_data = call_huggingface_api(request.prompt)
        
        # Extract the answer
        answer = api_data['choices'][0]['message']['content']
        return {"response": answer}

    except Exception as e:
        # If anything went wrong in the worker, we catch it here
        return {"error": str(e)}
