# ABOUTME: Thin command-line mirror that checks the constitution reached every target system message.
# ABOUTME: Usage: verify_target_preamble.py <log.eval> <preamble.md>; exit 0 only when all do.
import sys

from src.eval.audits.petri.tenet_audit.verify_target_preamble import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
