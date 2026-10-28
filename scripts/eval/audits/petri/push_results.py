# ABOUTME: Thin command-line mirror that publishes one finished tenet-audit run to the Hub.
# ABOUTME: Usage and flags: src/eval/audits/petri/tenet_audit/push_results.py.
import fire

from src.eval.audits.petri.tenet_audit.push_results import main

if __name__ == "__main__":
    fire.Fire(main)
