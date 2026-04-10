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
# MANDATORY: OPENAI CLIENT (STRICT PROXY)
# ==========================================
# The validator REQUIRES os.environ["API_BASE_URL"] and os.environ["API_KEY"]
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

# For local development ONLY (Will be overridden by platform env vars)
if not API_BASE_URL:
    API_BASE_URL = "https://api.openai.com/v1"
if not API_KEY:
    API_KEY = os.environ.get("HF_TOKEN", "missing-key")

# Ensure base_url does not end with /chat/completions (OpenAI client adds it)
if API_BASE_URL.endswith("/chat/completions"):
    API_BASE_URL = API_BASE_URL.replace("/chat/completions", "")

print(f"DEBUG: Using API_BASE_URL: {API_BASE_URL}")

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
    except Exception:
        pass

env = CropEnv(dataset)

# ==========================================
# UI & ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return f"🌿 API Active. Proxy: {API_BASE_URL}. Mode: {'Mock' if env.mock_mode else 'Real'}"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "proxy": API_BASE_URL})

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
    
    # ⚡ MANDATORY LLM CALL (Works in both Real and Mock mode)
    # This ensures the Validator observes API requests through the proxy
    try:
        # Determine target for advisory
        if not env.mock_mode and dataset:
            target = dataset.classes[int(action)]
        else:
            target = "a diseased crop"

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a crop advisor."},
                {"role": "user", "content": f"Give a VERY short treatment tip for {target}."}
            ],
            max_tokens=30
        )
        advisory = response.choices[0].message.content
    except Exception as e:
        advisory = f"Advisory unavailable (Proxy Error: {str(e)})"

    return jsonify({
        "state": {"step": 1, "done": done, "advisory": advisory},
        "reward": reward,
        "done": done
    })

def main():
    app.run(host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()