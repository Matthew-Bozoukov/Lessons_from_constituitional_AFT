# ABOUTME: One-off: print today's RunPod spend as a bare number (was the account balance
# ABOUTME: for the 2026-08-06 low-balance alarm; REST v2 exposes spend, not credit).

from src.infra import runpod

print(f"{runpod.account_spend()['spend_today_usd']:.2f}")
