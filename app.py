import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from env import CropEnv
from dataset import CropDataset
from models import get_model
import base64
import io

app = Flask(__name__)

# ==========================================
# OPENENV CONFIG
# ==========================================
NUM_CLASSES = 16
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "PlantVillage",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
    "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Initialize Dataset and Env
dataset = CropDataset("dataset/")
env = CropEnv(dataset)

# ==========================================
# OPENENV ENDPOINTS (MANDATORY)
# ==========================================

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json() or {}
    difficulty = data.get("difficulty", "easy")
    
    # Reset env and get first observation (tensor)
    obs = env.reset(difficulty=difficulty)
    
    # For JSON response, we might need a serializable state. 
    # OpenEnv usually expects the observation as "state".
    # If it's an image, we can return a reference or a small metadata.
    # However, to follow strict requirements:
    return jsonify({
        "state": {
            "step": 0,
            "message": "Environment reset successful",
            "difficulty": difficulty
        }
    })

@app.route("/step", methods=["POST"])
def step():
    data = request.get_json() or {}
    action = data.get("action")
    
    if action is None:
        return jsonify({"error": "Missing action"}), 400
    
    # Take step
    obs, reward, done, info = env.step(action)
    
    # Ensure reward is in [0, 1]
    # env.py returns 1.0 or -1.0. Let's fix that in env.py later or here.
    reward = max(0.0, float(reward))
    
    return jsonify({
        "state": {
            "step": env.dataset.data.index(env.current_image) if hasattr(env, 'current_image') else 0,
            "done": done
        },
        "reward": reward,
        "done": done
    })

@app.route("/state", methods=["GET"])
def state():
    return jsonify({
        "current_label": env.current_label,
        "is_done": False
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Standard Prediction Route (for UI)
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")
    
    # Logic to run model...
    # (Simplified for now to share the same model logic)
    return jsonify({"disease": "Healthy", "confidence": 0.95})

if __name__ == "__main__":
    # HF Spaces use port 7860 by default
    app.run(host="0.0.0.0", port=7860)