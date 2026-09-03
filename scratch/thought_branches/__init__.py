# ABOUTME: Thought Branches (arXiv 2510.27484) for agentic rollouts: study the DISTRIBUTION
# ABOUTME: over trajectories by resampling from branch points, not the single rollout we logged.

"""Thought Branches for ODCV-Bench.

The paper (Macar & Bogdan et al., 2025) makes one argument: a single chain-of-thought
tells you what a model *said*, never what *caused* what it did. To get causation you
resample the continuation from a chosen point and compare outcome distributions.

Porting that to ODCV needs one adaptation. The paper's scenarios are single-shot — one
CoT, one output — so "resample from sentence i" is just another completion. ODCV is an
agentic loop: assistant thought -> tool call -> tool result -> assistant thought ... The
environment is part of the trajectory, so a branch point splits a *transcript*, and what
happens after depends on a container's state, not only on the model.

That gives the two sampler backends in `sampler.py`:

  frozen  Replay the recorded tool results as the prefix, then sample ONE assistant step
          at the branch point. Measures the local ACTION distribution ("what would it
          have done here?"). No Docker, no environment; one model call per sample.
  live    Replay the whole scenario in its real container, forcing the recorded assistant
          turns up to the branch point via a proxy (`prefix_proxy.py`), then let the model
          free-run. Measures the true ODCV outcome distribution. Needs Docker + a served
          model.

Layout:
  trajectory.py    parse `messages_record.txt` -> Trajectory / Step
  segment.py       split assistant thoughts into stable-id Chunks (the resampling unit)
  taxonomy.py      ODCV-adapted function tags + the auto-labeller prompt
  label.py         LLM auto-labeller (OpenRouter) and a zero-cost lexical fallback
  embed.py         sentence embeddings + cosine similarity (the `T_j ~ S_i` test)
  sampler.py       Sampler protocol + FrozenEnvSampler + LiveEnvSampler
  prefix_proxy.py  OpenAI-compatible shim that forces a recorded prefix, then goes live
  metrics.py       counterfactual importance, resilience, counterfactual++, effect curves
  descriptive.py   offline good-vs-bad contrasts over already-logged rollouts (no GPU)
  report.py        figures + markdown mirrors

Promotion note: this is AI-written code and lives in `scratch/` per CLAUDE.md. If it
earns its keep, `trajectory/segment/taxonomy/embed/metrics` belong in
`src/eval/misalignment/odcv/branches/`; `descriptive` and `report` stay per-experiment.
"""

from scratch.thought_branches.trajectory import (  # noqa: F401
    Step,
    Trajectory,
    load_trajectory,
    parse_transcript,
)

__all__ = ["Step", "Trajectory", "load_trajectory", "parse_transcript"]
