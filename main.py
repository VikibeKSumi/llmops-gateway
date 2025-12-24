from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

app = FastAPI()

# 1. Define Input Structure
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "LLM Gateway is Live via HuggingFace Client"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    # 2. Get Token
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise HTTPException(status_code=500, detail="API Token is missing on server!")

    # 3. Initialize the Client (The "Official" way)
    # We use GPT-2 for the test because it's small and always online.
    client = InferenceClient(model="gpt2", token=api_token)

    try:
        # 4. Generate
        # text_generation returns the raw text directly
        response = client.text_generation(request.prompt, max_new_tokens=50)
        return {"response": response}

    except Exception as e:
        return {"error": str(e)}
