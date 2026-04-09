import os
import sys
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# Load local secrets from .env if it exists
load_dotenv()

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
# MANDATORY: OPENAI CLIENT (PROXIED)
# ==========================================
# Requirements: Must use API_BASE_URL and API_KEY from environment
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "missing-key") # The platform will inject this

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

# ==========================================
# OPENENV CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
dataset = None

if os.path.exists(DATASET_PATH) and os.listdir(DATASET_PATH):
    try:
        dataset = CropDataset(DATASET_PATH)
        print("✅ Dataset loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading dataset: {e}")
else:
    print("ℹ️ Dataset missing. Mock Mode Active.")

env = CropEnv(dataset)

# ==========================================
# UI & HEALTH ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    mode = "MOCK MODE" if env.mock_mode else "REAL MODE"
    return f"🌿 Crop Disease API Active (Mode: {mode}). Using Proxy: {API_BASE_URL}"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "proxy": API_BASE_URL})

# ==========================================
# OPENENV ENDPOINTS (MANDATORY)
# ==========================================

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True, force=True) or {}
    difficulty = data.get("difficulty", "easy")
    env.reset(difficulty=difficulty)
    return jsonify({
        "state": {
            "step": 0,
            "mock": env.mock_mode
        }
    })

@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(silent=True, force=True) or {}
    action = data.get("action")
    if action is None:
        return jsonify({"error": "Missing action"}), 400
    
    obs, reward, done, info = env.step(action)
    
    # ⚡ LLM ADVISORY STEP (Mandatory for Proxy Tracking)
    # We use the LLM to provide advice based on the predicted action
    advisory = ""
    if not env.mock_mode:
        try:
            disease = dataset.classes[int(action)]
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a crop advisor."},
                    {"role": "user", "content": f"Provide a single treatment tip for {disease}."}
                ],
                max_tokens=50
            )
            advisory = response.choices[0].message.content
        except Exception as e:
            advisory = f"Proxy Call Failed: {str(e)}"

    return jsonify({
        "state": {"step": 1, "done": done, "advisory": advisory},
        "reward": reward,
        "done": done
    })

def main():
    app.run(host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()