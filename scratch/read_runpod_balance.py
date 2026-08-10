# ABOUTME: One-off: print the RunPod account balance as a bare number (used by the
# ABOUTME: low-balance alarm watching the 2026-08-06 self-reflection training pod).

import os

import requests
from dotenv import load_dotenv

load_dotenv()

r = requests.post(
    "https://api.runpod.io/graphql",
    json={"query": "query { myself { clientBalance } }"},
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
    timeout=30)
print(f"{r.json()['data']['myself']['clientBalance']:.2f}")
