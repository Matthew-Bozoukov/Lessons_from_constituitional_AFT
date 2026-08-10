#!/usr/bin/env python3
# ABOUTME: THE eval entrypoint; thin shim over src.eval.run_eval.main (also exposed as
# ABOUTME: `uv run run_eval` via [project.scripts]).
from src.eval.run_eval import main

if __name__ == "__main__":
    main()
