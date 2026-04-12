import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import io

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 1. CNN CONFIG & MODEL LOAD
# -----------------------------
NUM_CLASSES = 16
classes = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "PlantVillage",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
    "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
    "Tomato_healthy"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# 2. LLM INTEGRATION (PROXIED)
# -----------------------------
from openai import OpenAI

# Robust Proxy Configuration (Prioritize platform vars)
PROXY_BASE_URL = (
    os.environ.get("API_BASE_URL") or 
    os.environ.get("OPENAI_API_BASE") or 
    os.environ.get("OPENAI_BASE_URL") or 
    "https://api.openai.com/v1"
)
PROXY_API_KEY = (
    os.environ.get("API_KEY") or 
    os.environ.get("OPENAI_API_KEY") or 
    os.environ.get("HF_TOKEN") or
    "missing-key"
)

# Strip /chat/completions from the base URL if present
if PROXY_BASE_URL.endswith("/chat/completions"):
    PROXY_BASE_URL = PROXY_BASE_URL.replace("/chat/completions", "")
elif PROXY_BASE_URL.endswith("/chat/completions/"):
    PROXY_BASE_URL = PROXY_BASE_URL.replace("/chat/completions/", "")

print(f"🌿 CLOUD_INIT: Using Base URL: {PROXY_BASE_URL}")
print(f"🌿 CLOUD_INIT: API Key Detected: {bool(PROXY_API_KEY) and PROXY_API_KEY != 'missing-key'}")

client = OpenAI(
    base_url=PROXY_BASE_URL,
    api_key=PROXY_API_KEY
)

HF_MODEL = os.environ.get("MODEL_NAME", "gpt-4o")

def call_qwen_llm(disease, task):
    if not PROXY_API_KEY:
        return "AI Advisory currently unavailable (Missing API Token)."

    if task == "medium":
        prompt = f"You are an agricultural expert. Give a short description and basic cure for {disease}. Use simple language and bullet points."
    elif task == "hard":
        prompt = f"You are an agricultural expert. Provide detailed information about {disease}. Include: - Description, - Causes, - Treatment (bullet points), - Prevention tips, - Extra insights. Keep it practical for farmers."
    else:
        return ""

    try:
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000 if task == "hard" else 300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Consulting our agricultural database... (LLM Error: {str(e)})"

# -----------------------------
# 3. API ENDPOINTS
# -----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...), task: str = Form("easy")):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # CNN Inference
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs).item()
            confidence = torch.max(probs).item()

        disease_name = classes[pred_idx].replace("__", " ").replace("_", " ")
        
        # Build Response
        response = {
            "disease": disease_name,
            "confidence": round(confidence, 4),
            "task": task
        }

        # Multi-Level Logic
        if task == "easy":
            # Task 1: Disease Name only (already in base response)
            pass
        elif task in ["medium", "hard"]:
            # Task 2 & 3: Fetch LLM data
            advisory = call_qwen_llm(disease_name, task)
            response["advisory"] = advisory
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "CropAI Backend is Active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
