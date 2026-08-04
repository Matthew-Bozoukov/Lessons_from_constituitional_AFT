# Upstream assets — ai-psychosis

Verbatim copies of the scientific inputs from Tim Hua's AI-psychosis red-teaming
study. These files are the eval's rubric and stimuli; the harness around them is
our own native reimplementation (see the modules one level up), so nothing else
from the upstream repo is vendored.

- **Upstream**: https://github.com/tim-hua-01/ai-psychosis
  (MIT — see `LICENSE` in this directory), pinned at commit
  `b6593b38b18edf0b9267784c1f3e9357c6a6be6e` (copied 2026-08-04).
- **Write-up**: "AI-Induced Psychosis: A shallow investigation",
  https://www.lesswrong.com/posts/iGF7YcnQkEbwvYLPA/ai-induced-psychosis-a-shallow-investigation

| file | role |
| --- | --- |
| `red_team_prompt.txt` | Instructions for the persona-playing red-teamer. Placeholders `{character_name}` / `{psychotic_character}` are filled per persona. |
| `grader_prompt.txt` | The judge's rubric: 5 core questions (referral, pushback, delusion confirmation + quotes) and 9 CBT-manual criteria, returned as one flat 14-key JSON object. |
| `characters/*.txt` | Nine personas, each a delusion scenario with a 12-turn progression arc. Character name = first token of the filename, title-cased (`ethan_reality.txt` → Ethan). |

Do not edit these in place — they are the comparison point against upstream's
published results. If upstream revises them, re-copy and update the pinned SHA
here.
