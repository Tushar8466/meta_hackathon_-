
Copy

import os
import sys
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
 
# Load local .env ONLY if platform vars are not already set
load_dotenv(override=False)
 
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
# MANDATORY: OPENAI CLIENT (STRICT PROXY)
# ==========================================
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "missing-key")
 
# Ensure base_url does not end with /chat/completions (OpenAI client adds it)
if API_BASE_URL.endswith("/chat/completions"):
    API_BASE_URL = API_BASE_URL.replace("/chat/completions", "")
 
print(f"DEBUG: Using API_BASE_URL: {API_BASE_URL}")
print(f"DEBUG: API_KEY present: {bool(API_KEY and API_KEY != 'missing-key')}")
 
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
    except Exception as e:
        print(f"DEBUG: Dataset load failed: {e}")
 
env = CropEnv(dataset)
 
# ==========================================
# ROUTES
# ==========================================
 
@app.route("/", methods=["GET"])
def home():
    return f"🌿 API Active. Proxy: {API_BASE_URL}. Mode: {'Mock' if env.mock_mode else 'Real'}"
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "proxy": API_BASE_URL,
        "model": MODEL_NAME,
        "api_key_present": bool(API_KEY and API_KEY != "missing-key")
    })
 
@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True, force=True) or {}
    difficulty = data.get("difficulty", "easy")
    env.reset(difficulty=difficulty)
    return jsonify({"state": {"step": 0, "mock": env.mock_mode}})
 
@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(silent=True, force=True) or {}
    action = data.get("action")
 
    if action is None:
        return jsonify({"error": "Missing action"}), 400
 
    obs, reward, done, info = env.step(action)
 
    # ==========================================
    # MANDATORY LLM CALL THROUGH PROXY
    # ==========================================
    try:
        # Determine target crop/disease for advisory
        if not env.mock_mode and dataset:
            target = dataset.classes[int(action)]
        else:
            target = "a diseased crop"
 
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a crop disease advisor. Be concise."},
                {"role": "user", "content": f"Give a very short treatment tip for {target}."}
            ],
            max_tokens=50
        )
        advisory = response.choices[0].message.content
 
    except Exception as e:
        print(f"DEBUG: LLM call failed: {e}")
        advisory = f"Advisory unavailable: {str(e)}"
 
    return jsonify({
        "state": {
            "step": 1,
            "done": done,
            "advisory": advisory
        },
        "reward": reward,
        "done": done
    })
 
 
def main():
    app.run(host="0.0.0.0", port=7860)
 
 
if __name__ == "__main__":
    main()