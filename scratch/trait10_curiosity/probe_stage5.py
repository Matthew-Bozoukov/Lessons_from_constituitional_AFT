# ABOUTME: Replay stage 5 (draft_responses) of the t10 config for one record against the
# ABOUTME: configured model and print the whole reply, so a tag-format failure can be read.
# Run: uv run python scratch/trait10_curiosity/probe_stage5.py --run_dir <smoke dir> [--idx 1]

import json
import re
from pathlib import Path

import fire
from omegaconf import OmegaConf

from src.data.synth.constitution import units_from_config
from src.endpoints.openrouter import OpenRouterClient

CFG = "scratch/trait10_curiosity/difficult_advice_t10.yaml"


def main(run_dir: str, idx: int = 1, model: str | None = None,
         max_tokens: int | None = None, reasoning: str | None = None) -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(CFG), resolve=True)
    units, style = units_from_config(cfg)
    trait = units[0].as_trait()
    rows = [json.loads(l) for l in open(Path(run_dir) / "stage_5_revise_prompts.jsonl")]
    r = rows[idx]
    stage = next(s for s in cfg["stages"] if s["name"] == "draft_responses")
    m = cfg["models"][stage["model"]]
    fields = {**r, "trait_name": trait.name, "trait_text": trait.text,
              "style_guidance": style}
    messages = [{"role": "system", "content": stage["prompts"]["system"].format(**fields)},
                {"role": "user", "content": stage["prompts"]["user"].format(**fields)}]
    kwargs = {}
    if reasoning:
        kwargs["extra_body"] = {"reasoning": json.loads(reasoning)}
    res = OpenRouterClient().chat(model or m["model"], messages,
                                  temperature=float(m["temperature"]),
                                  max_tokens=int(max_tokens or m["max_tokens"]), **kwargs)
    text = res.content
    print(f"model={model or m['model']} provider={res.provider} finish={res.finish_reason} "
          f"prompt_tokens={res.prompt_tokens} completion_tokens={res.completion_tokens} "
          f"chars={len(text)}")
    for k in ("reasoning", "response"):
        print(f"  <{k}>: open={text.count(f'<{k}>')} close={text.count(f'</{k}>')}")
    print("---- HEAD ----")
    print(text[:1200])
    print("---- TAIL ----")
    print(text[-1500:])
    print("---- other tags seen ----", sorted(set(re.findall(r"</?([a-zA-Z_]+)>", text))))


if __name__ == "__main__":
    fire.Fire(main)
