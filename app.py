import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from env import CropEnv
from dataset import CropDataset

app = Flask(__name__)

# ==========================================
# OPENENV CONFIG
# ==========================================
DATASET_PATH = "dataset"
dataset = None

# Only try to load dataset if the folder exists and is not empty
if os.path.exists(DATASET_PATH) and os.listdir(DATASET_PATH):
    try:
        dataset = CropDataset(DATASET_PATH)
        print("✅ Dataset loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading dataset, switching to mock mode: {e}")
else:
    print("ℹ️ Dataset folder missing or empty. Using MOCK MODE for OpenEnv.")

# Initialize Env (will auto-switch to mock if dataset is None)
env = CropEnv(dataset)

# ==========================================
# OPENENV ENDPOINTS (MANDATORY)
# ==========================================

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json() or {}
    difficulty = data.get("difficulty", "easy")
    env.reset(difficulty=difficulty)
    return jsonify({
        "state": {
            "step": 0,
            "message": "Environment reset successful (Mock Mode Active)" if env.mock_mode else "Environment reset successful",
            "mock": env.mock_mode
        }
    })

@app.route("/step", methods=["POST"])
def step():
    data = request.get_json() or {}
    action = data.get("action")
    if action is None:
        return jsonify({"error": "Missing action"}), 400
    
    obs, reward, done, info = env.step(action)
    return jsonify({
        "state": {"step": 1, "done": done},
        "reward": reward,
        "done": done
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mock_mode": env.mock_mode})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)