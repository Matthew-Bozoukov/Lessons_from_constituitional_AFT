# Upstream assets — Capture the Fish (ctfish)

Verbatim copies of the two scientific inputs of Palisade Research's chess specification-gaming
environment: the game script the agent plays against (`env/game.py`) and the escalation-scale
LLM judge prompt (`judge.txt`). The agent harness around them is our own reimplementation —
upstream's `player.py` is NOT vendored; its prompts are transcribed verbatim into
`../prompts.py` and its OODA state machine reimplemented in `../agent.py`, because the loop
runs in the eval driver here rather than inside the container (see `../agent.py` for why).

- **Upstream**: https://github.com/PalisadeResearch/ctfish, pinned at commit
  `f029c613ec40b059cc8d4742cb515a138aa6922c` (2025-08-26, copied 2026-09-04).
  - `env/game.py` = upstream `game.py`, byte-identical
    (sha256 `65c27f39f44ab2d8db322e288b19c5462acf5d36b5d92a209e1884f64b7bc1da`).
  - `judge.txt` = upstream `scoring/hacking/judge.txt`, byte-identical
    (sha256 `f3403193f4636ee6665012b94609a68827fa2e6574f74cf9b9a54768ad61de1a`).
- **Paper**: Bondarenko, Volk, Volkov, Ladish. "Demonstrating specification gaming in
  reasoning models." arXiv:2502.13295.

`env/Dockerfile` is OURS, not upstream's: upstream's image bakes in the agent harness and
its API clients, and this one is a bare sandbox (stockfish + `game.py` + a shell) because
the harness runs in the driver. What the agent can see and touch inside `/env` is the same.

## Licence status — READ BEFORE REDISTRIBUTING

**The upstream repository publishes no licence** (no `LICENSE` file, no licence field, no
statement in the README — verified at the pinned commit). Default copyright therefore
applies to upstream's contribution.

These files are vendored for internal research reproducibility: the environment is only
meaningful if the exact game script and rubric are pinned, and a runtime fetch would make
our results depend on a third-party repository staying available and unchanged.

Do **not** redistribute this directory outside the project — in particular, do not include
`assets/` in any Hugging Face upload. `../runner.py` publishes rollout transcripts, judge
verdicts and scores, never these files, for exactly that reason.
