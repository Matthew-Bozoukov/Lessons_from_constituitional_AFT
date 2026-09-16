# ABOUTME: Colosseum multi-agent collusion eval — six LLM agents on a shared Jira task,
# ABOUTME: measured by regret against the computable optimum rather than by reading transcripts.

"""Does constitutional SFT change what a model does on a TEAM?

Constitutional-SFT results — Anthropic's *Teaching Claude Why*, the GDM synthetic-document
traits work, and this repo's own difficult-advice arms — are all trained and measured with
one model acting alone. Hammond et al. 2025 (*Multi-Agent Risks from Advanced AI*,
arXiv 2502.14143) argue that single-agent safety does not imply multi-agent safety, and
name collusion between agents as one of three core failure modes. Nobody has measured
whether constitutional training changes collusion, or whether its effect is smaller in a
team than alone. This package measures both.

The environment is Colosseum's Jira task (Nakamura et al. 2026, arXiv 2602.15198), built
on Terrarium: six agents divide tickets to maximise a shared joint reward, first by
messaging each other and then by assigning tickets with tool calls. Because the task is a
distributed constraint optimisation problem, the best achievable outcome is COMPUTABLE —
so misbehaviour is a measured drop in outcome (regret), not a judge's reading of what the
agents said. The judge score is kept as a secondary measure precisely because the paper
found talk and action diverge: models collude on paper without acting on it, and act
without saying so.
"""
