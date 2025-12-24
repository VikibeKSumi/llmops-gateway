from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# 1. Setup the Model Link
# We use GPT-2 because it is the "Hello World" of LLMs (always online)
API_URL = "https://router.huggingface.co/hf-inference/models/gpt2"

# 2. Define Input Structure
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 100 # Optional limit

@app.get("/")
def home():
    return {"message": "LLM Gateway is Live. Use /generate to chat."}

@app.post("/generate")
def generate_text(request: PromptRequest):
    # DEBUGGING: Print all available keys to the system logs
    print("Available Env Vars:", os.environ.keys()) 
    api_token = os.environ.get("HF_TOKEN")
    
    # 3. Get the Secret Key securely
    api_token = os.environ.get("HF_TOKEN")
    
    if not api_token:
        raise HTTPException(status_code=500, detail="API Token is missing on server!")

    # 4. Prepare the Payload for Hugging Face
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": request.prompt,
        "parameters": {
            "max_new_tokens": request.max_length,
            "return_full_text": False, # Only return the new answer, not the prompt
            "temperature": 0.7 # Creativity level
        }
    }

    # 5. Send to the "Brain"
    response = requests.post(API_URL, headers=headers, json=payload)

    # 6. Handle Response
    if response.status_code != 200:
        return {"error": f"Hugging Face Error: {response.text}"}
    
    # Hugging Face returns a list of dictionaries
    result = response.json()
    generated_text = result[0]['generated_text']
    
    return {"response": generated_text}
