# ABOUTME: Prints the RunPod account balance and the OpenRouter credit left (no secrets printed).
# ABOUTME: Run: uv run python scratch/qwen38/balances.py
import os

import requests
from dotenv import load_dotenv

load_dotenv()
r = requests.post(
    "https://api.runpod.io/graphql",
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
    json={"query": "{ myself { clientBalance currentSpendPerHr } }"},
    timeout=30,
)
me = r.json()["data"]["myself"]
print(
    f"RunPod balance ${me['clientBalance']:.2f}, spending ${me['currentSpendPerHr']:.2f}/h"
)
o = requests.get(
    "https://openrouter.ai/api/v1/credits",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    timeout=30,
).json()["data"]
print(f"OpenRouter credit left ${o['total_credits'] - o['total_usage']:.2f}")
