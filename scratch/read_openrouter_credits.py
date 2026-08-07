# ABOUTME: One-off: print the OpenRouter credit balance (docs/EXPENDITURE.md requires a
# ABOUTME: before/after /credits reading around every paid generation run).

import os

import requests
from dotenv import load_dotenv

load_dotenv()

r = requests.get("https://openrouter.ai/api/v1/credits",
                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                 timeout=30)
d = r.json()["data"]
print(f"credits: {d['total_credits']:.2f}  usage: {d['total_usage']:.2f}  "
      f"remaining: {d['total_credits'] - d['total_usage']:.2f}")
