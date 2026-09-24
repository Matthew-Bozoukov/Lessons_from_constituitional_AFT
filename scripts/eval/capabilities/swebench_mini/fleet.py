# ABOUTME: Thin CLI mirror for the durable SWE-bench Lite campaign driver.
# ABOUTME: Prefer uv run evals --name swebench_mini --fleet for a new target.
from src.eval.capabilities.swebench_mini.fleet import main

if __name__ == '__main__':
    main()
