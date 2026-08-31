# ABOUTME: Regenerate the agentic-misalignment arm comparison including the new
# ABOUTME: t2-9000-synthdoc-1000 inspect arm, baselined against table2-only.
from scratch.compare_am_arms import main

ARMS = {
    "t2+synthdoc (2203)": "output/inspect_agentic_misalignment/table2_synthdoc_r64/direct",
    "t2+synthdoc (1000)": "output/inspect_agentic_misalignment/t2synth1000_125ep",
    "t2+synthdoc (716)": "output/inspect_agentic_misalignment/t2716_125ep",
    "t2+memself (2000)": "output/inspect_agentic_misalignment/memself_125ep",
    "table2-only": "output/inspect_agentic_misalignment/table2_only_9284/direct",
}

if __name__ == "__main__":
    main(arms=ARMS, baseline="table2-only")
