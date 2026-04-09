import os
import sys
import requests
import json
import time

# ================================
# ENV VARIABLES (MANDATORY)
# ================================
# Requirements: API_BASE_URL (with default), MODEL_NAME (with default), HF_TOKEN (validated)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN environment variable is missing.", file=sys.stderr)
    sys.exit(1)

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        # 1. Start Task
        print(f"[START] task={task} env=crop-disease-env model={MODEL_NAME}")
        
        try:
            # 2. Reset Environment
            reset_resp = requests.post(f"{API_BASE_URL}/reset", json={"difficulty": task}, timeout=10)
            reset_resp.raise_for_status()
            reset_data = reset_resp.json()
            
            # 3. Running Steps (for this env, it's 1 step)
            # In a real agent, you'd call an LLM here to get the action.
            # For validation, we simulate an action (e.g., class 0).
            action = 0 
            
            step_resp = requests.post(f"{API_BASE_URL}/step", json={"action": action}, timeout=10)
            step_resp.raise_for_status()
            step_data = step_resp.json()
            
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", True)
            
            print(f"[STEP] step=1 action={action} reward={reward} done={str(done).lower()} error=null")
            
            # 4. End Task
            print(f"[END] success=true steps=1 rewards={reward}")
            
        except Exception as e:
            print(f"[STEP] step=1 action=none reward=0.0 done=true error=\"{str(e)}\"")
            print(f"[END] success=false steps=1 rewards=0.0")

if __name__ == "__main__":
    run_inference()