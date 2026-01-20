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
# Change this import at the top
from langsmith import traceable

@traceable
def call_groq_api(user_prompt: str):
    # 1. Get the Key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API Key is missing!")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 2. The Groq Endpoint (Standard OpenAI format)
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # 3. Payload: Using Llama-3.3 (State of the Art)
    payload = {
        "model": "llama-3.3-70b-versatile", 
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Groq Error {response.status_code}: {response.text}")
        
    return response.json()

# --- IMPORTANT: Update your /generate endpoint to call this new function ---
# 3. The Endpoint (The "Manager")
# It just receives the request and delegates to the worker.
@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        # CRITICAL: We call the decorated function here!
        # This triggers the LangSmith trace.
        # Call Groq instead of HF
        api_data = call_groq_api(request.prompt)
        
        # The structure is the same!
        # Extract the answer
        answer = api_data['choices'][0]['message']['content']
        return {"response": answer}

    except Exception as e:
        # If anything went wrong in the worker, we catch it here
        return {"error": str(e)}
