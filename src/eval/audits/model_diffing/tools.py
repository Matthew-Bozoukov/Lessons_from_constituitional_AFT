# ABOUTME: The diffing agent's one real tool: sample both models on the same prompts and
# ABOUTME: return the pairs with within-model spread made visible.

"""Paired sampling, with variance in front of the agent.

The failure mode this guards against is the agent concluding "A says X, B says Y" from one
sample each, when both models would have said either at temperature 1.0. Every prompt is
therefore sampled n>=5 times per model, and the tool result interleaves the samples so the
agent can see within-model disagreement before it compares across models.

The two "models" are just two served names on one vLLM process, so serving is identical by
construction — for the A-vs-A control they are literally the same adapter.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

MIN_SAMPLES = 5
MAX_PROMPTS = 5
# Long thinking answers would swamp the agent's context in a couple of turns; it needs
# enough to judge style and substance, not the whole essay.
SAMPLE_CHARS = 1200


class PairedSampler:
    """Samples two served model names on identical prompts."""

    def __init__(self, base_url: str, model_a: str, model_b: str,
                 temperature: float = 1.0, max_tokens: int = 3072,
                 workers: int = 20) -> None:
        """Initialise the sampler.

        Args:
            base_url: OpenAI-compatible endpoint serving both names.
            model_a: Served name presented to the agent as Model A.
            model_b: Served name presented to the agent as Model B.
            temperature: Sampling temperature; 1.0 so within-model variance is real.
            max_tokens: Per-sample cap.
            workers: Thread pool size for the fan-out.
        """
        self.client = OpenAI(base_url=base_url, api_key="dummy", timeout=300, max_retries=3)
        self.model_a, self.model_b = model_a, model_b
        self.temperature, self.max_tokens, self.workers = temperature, max_tokens, workers
        self.call_count = 0

    def _one(self, model: str, prompt: str) -> str:
        """One completion; returns the answer text or an error marker."""
        try:
            r = self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens, temperature=self.temperature)
            text = (r.choices[0].message.content or "").strip()
            # A sample cut off at the token cap must say so: an agent shown a truncated
            # answer beside a complete one will read the difference as a model difference.
            # 17% of samples hit the cap on the first control run.
            if r.choices[0].finish_reason == "length":
                text += " [CUT OFF AT TOKEN LIMIT — not a complete answer]"
            return text
        except Exception as e:  # noqa: BLE001 - a dead sample must not kill the turn
            return f"[SAMPLE FAILED: {type(e).__name__}]"

    def send_messages(self, prompts: list[str], n_samples: int = MIN_SAMPLES) -> str:
        """Sample both models on each prompt and format the pairs for the agent.

        Args:
            prompts: Up to MAX_PROMPTS prompts to send to both models.
            n_samples: Samples per model per prompt; floored at MIN_SAMPLES so a
                single-sample comparison is impossible by construction.

        Returns:
            A formatted transcript block, A and B interleaved per prompt.
        """
        prompts = prompts[:MAX_PROMPTS]
        n = max(int(n_samples), MIN_SAMPLES)
        jobs = [(model, p) for p in prompts for model in (self.model_a, self.model_b)
                for _ in range(n)]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(lambda j: self._one(*j), jobs))
        self.call_count += len(jobs)

        out, k = [], 0
        for p in prompts:
            out.append(f"\n{'=' * 70}\nPROMPT: {p}\n{'=' * 70}")
            for label in ("A", "B"):
                out.append(f"\n--- Model {label} ({n} samples) ---")
                for i in range(n):
                    text = results[k][:SAMPLE_CHARS]
                    trunc = "…[truncated]" if len(results[k]) > SAMPLE_CHARS else ""
                    out.append(f"[{label}{i + 1}] {text}{trunc}")
                    k += 1
        out.append(
            f"\n(Each model answered every prompt {n} times at temperature "
            f"{self.temperature}. Differences smaller than the spread WITHIN a model are "
            "not evidence of a difference between them.)")
        return "\n".join(out)
