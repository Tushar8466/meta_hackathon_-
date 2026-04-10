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
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

try:
    from env import CropEnv
    from dataset import CropDataset
except ImportError:
    # Fallback if structure is different
    from .env import CropEnv
    from .dataset import CropDataset

app = Flask(__name__)

# ==========================================
# MANDATORY: OPENAI CLIENT (STRICT PROXY)
# ==========================================
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", ""))

# Strip '/chat/completions' if present, as the OpenAI client appends it automatically
if API_BASE_URL.endswith("/chat/completions"):
    API_BASE_URL = API_BASE_URL.replace("/chat/completions", "")
elif API_BASE_URL.endswith("/chat/completions/"):
    API_BASE_URL = API_BASE_URL.replace("/chat/completions/", "")

# Ensure it ends with /v1 if it's hitting a standard API and doesn't have it
# But don't force it if it's a custom proxy (some proxies prefer no /v1)

print(f"DEBUG: Using API_BASE_URL: {API_BASE_URL}")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY if API_KEY else "missing-key"
)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

# ==========================================
# OPENENV CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Dataset is usually in the parent or current dir
DATASET_PATH = os.path.join(PARENT_DIR, "dataset")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(CURRENT_DIR, "dataset")

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
    return f"🌿 CropAI API Active. Proxy: {API_BASE_URL}. Mode: {'Mock' if env.mock_mode else 'Real'}"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "proxy": API_BASE_URL,
        "model": MODEL_NAME,
        "api_key_present": bool(API_KEY and API_KEY != "missing-key"),
        "mock_mode": env.mock_mode
    })

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True, force=True) or {}
    difficulty = data.get("difficulty", "easy")
    obs = env.reset(difficulty=difficulty)
    
    # Convert tensor to list for JSON serialization
    if isinstance(obs, torch.Tensor):
        obs = obs.tolist()
        
    return jsonify({
        "observation": obs,
        "info": {"mode": "mock" if env.mock_mode else "real"}
    })

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
            try:
                target = dataset.classes[int(action)]
            except:
                target = "a diseased crop"
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

    # Convert tensor to list for JSON serialization
    if isinstance(obs, torch.Tensor):
        obs = obs.tolist()

    return jsonify({
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": {
            "advisory": advisory,
            "step": 1
        }
    })

if __name__ == "__main__":
    # HF Spaces expect port 7860
    app.run(host="0.0.0.0", port=7860)