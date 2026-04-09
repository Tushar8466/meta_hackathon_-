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
# 2. LLM INTEGRATION (Hugging Face)
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("VITE_HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
# Using the specific provider requested: featherless-ai
HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai"

def call_qwen_llm(disease, task):
    if not HF_TOKEN:
        return "AI Advisory currently unavailable (Missing API Token)."

    if task == "medium":
        prompt = f"You are an agricultural expert. Give a short description and basic cure for {disease}. Use simple language and bullet points."
    elif task == "hard":
        prompt = f"You are an agricultural expert. Provide detailed information about {disease}. Include: - Description, - Causes, - Treatment (bullet points), - Prevention tips, - Extra insights. Keep it practical for farmers."
    else:
        return ""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000 if task == "hard" else 300,
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
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
