from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import traceback  # <--- New Tool for debugging

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
        # We try to connect to GPT-2
        client = InferenceClient(model="gpt2", token=api_token)
        
        # We try to generate text
        response = client.text_generation(request.prompt, max_new_tokens=50)
        return {"response": response}

    except Exception as e:
        # <--- THIS IS THE FIX --->
        # We capture the full technical name of the error and the line number
        error_name = repr(e) 
        full_traceback = traceback.format_exc()
        
        # We print it to the Render logs
        print("CRITICAL ERROR:", full_traceback)
        
        # We send it back to you in the API
        return {
            "error_type": error_name,
            "details": full_traceback
        }
