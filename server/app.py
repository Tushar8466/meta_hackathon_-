import os
import sys
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# ==========================================
# PATH INJECTION (Ensures imports work everywhere)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from env import CropEnv
from dataset import CropDataset

app = Flask(__name__)

# ==========================================
# OPENENV CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
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
# UI & HEALTH ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    mode = "MOCK MODE (Simulated)" if env.mock_mode else "REAL MODE (Dataset Loaded)"
    return f"""
    <html>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1 style="color: #27ae60;">🌿 Crop Disease API is Active</h1>
            <p><strong>Status:</strong> Running in {mode}</p>
            <p>OpenEnv Endpoints: <code>/reset</code>, <code>/step</code></p>
        </body>
    </html>
    """

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mock_mode": env.mock_mode})

# ==========================================
# OPENENV ENDPOINTS (MANDATORY)
# ==========================================

@app.route("/reset", methods=["POST"])
def reset():
    # Use silent=True and force=True to be tolerant of mission Content-Type headers
    data = request.get_json(silent=True, force=True) or {}
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
    # Use silent=True and force=True to be tolerant of mission Content-Type headers
    data = request.get_json(silent=True, force=True) or {}
    action = data.get("action")
    if action is None:
        return jsonify({"error": "Missing action"}), 400
    
    obs, reward, done, info = env.step(action)
    return jsonify({
        "state": {"step": 1, "done": done},
        "reward": reward,
        "done": done
    })

def main():
    # HF Spaces use port 7860 by default
    app.run(host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()