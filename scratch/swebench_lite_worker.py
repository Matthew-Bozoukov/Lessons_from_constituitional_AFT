# ABOUTME: Compatibility entry point for the supported SWE-bench Lite fleet implementation.
# ABOUTME: Existing commands and tests delegate to the canonical module without duplicating logic.
import sys
from src.eval.capabilities.swebench_mini import fleet_worker as implementation
if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
