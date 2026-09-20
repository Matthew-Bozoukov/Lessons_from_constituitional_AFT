# ABOUTME: The one keyword rule for "this text is about other AI agents", shared by count_multiagent.py
# ABOUTME: and splice_corpus.py's gate. "AI" alone never counts: every row is about an AI assistant.
"""Widened 2026-09-20 after reading 12 unmatched scenarios of the steered run: 7 were
multi-agent in words the first rule lacked ("peer instances", "peer systems", "four identical
instances", "three robotic inspection units"); the other 5 had one AI and no other agent.
"""

import re

_KIND = r"(AIs?|agents?|models?|instances?|systems?|assistants?|bots?|coordinators?|moderators?|units?|copies|copy)"
_MANY = r"(multiple|several|two|three|four|five|six|seven|eight|nine|ten|dozens of|many|identical|parallel|federated)"

MULTI = re.compile(
    r"\b(sub-?agents?|orchestrat\w+|multi-?agent"
    rf"|(other|another|peer|fellow|partner|sibling|rival|second|third|specialist|worker|downstream|upstream) (AI )?{_KIND}"
    rf"|{_MANY} (other |identical |peer |AI |robotic |autonomous )*{_KIND}"
    r"|agents? (team|swarm|system|network|pipeline)"
    r"|(instances?|cop(y|ies)) of (itself|yourself|myself|the (assistant|model|AI|same))"
    r"|AI-to-AI|agent-to-agent|negotiating (AI|agent|bot)|automated pipeline|many hands)\b",
    re.IGNORECASE,
)
