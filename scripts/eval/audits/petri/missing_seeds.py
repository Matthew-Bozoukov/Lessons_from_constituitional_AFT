# ABOUTME: Thin command-line mirror that collects the tenet-audit seeds still lacking a valid score.
# ABOUTME: Usage and flags: src/eval/audits/petri/tenet_audit/missing_seeds.py.
import fire

from src.eval.audits.petri.tenet_audit.missing_seeds import main

if __name__ == "__main__":
    fire.Fire(main)
