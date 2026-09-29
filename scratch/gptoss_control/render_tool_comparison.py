# ABOUTME: Render the same ODCV tools with pinned Qwen, historical Harmony, and current Harmony.
# ABOUTME: Offline prompt inspection only; no model sampling, tool execution, or credentials.
from __future__ import annotations

import ast
import subprocess
import types
from pathlib import Path

from transformers import AutoTokenizer

from src.infra.endpoints import harmony
from src.model_profile import render_chat


def main():
    bench = Path("src/eval/misalignment/odcv/third_party/odcv-bench/mission_executor/agent_main.py")
    tree = ast.parse(bench.read_text(encoding="utf-8"))
    tools = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "BASE_TOOLS"
                         for target in node.targets))
    messages = [{"role": "system", "content": "You are an agent completing a task."},
                {"role": "user", "content": "Inspect the files."}]
    previous = types.ModuleType("historical_harmony")
    previous_revision = "87957e51"
    source = subprocess.check_output(
        ["git", "show", f"{previous_revision}:src/infra/endpoints/harmony.py"], text=True)
    exec(compile(source, "historical_harmony.py", "exec"), previous.__dict__)
    renderer = harmony.make_renderer(local_files_only=True)
    qwen_revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", revision=qwen_revision,
                                        local_files_only=True)
    prompts = {
        "Qwen 3.6 27B": render_chat(qwen, messages, tools=tools,
                                   render_kwargs={"preserve_thinking": True},
                                   add_generation_prompt=True),
        "GPT-OSS before": renderer.tokenizer.decode(previous.render_prompt(renderer, messages, tools).to_ints()),
        "GPT-OSS after": renderer.tokenizer.decode(harmony.render_prompt(renderer, messages, tools).to_ints()),
    }
    # Definitions and task text must survive unchanged; only presentation changes.
    for prompt in prompts.values():
        assert "Inspect the files." in prompt
        assert "task_complete" in prompt
        assert tools[0]["function"]["description"] in prompt
    assert "\nJSON Schema for arguments:" in prompts["GPT-OSS before"]
    assert "\nJSON Schema for arguments:" not in prompts["GPT-OSS after"]
    assert "\n// JSON Schema for arguments:" in prompts["GPT-OSS after"]
    assert harmony.TOOL_FORMAT_INSTRUCTIONS in prompts["GPT-OSS after"]
    text = """<!-- ABOUTME: Exact rendered tool-prompt comparison for the September 29 Harmony update. -->
<!-- ABOUTME: Offline template inspection, not sampled model outputs or evidence of improved accuracy. -->

# Tool-prompt comparison, 2026-09-29

The same two ODCV tools (`bash` and `task_complete`) and the same example conversation
are passed through all three renderers. These are actual decoded prompts, not model
responses. Qwen's native template provides XML-call instructions and an example.
The Harmony update adds JSON-call instructions and an ODCV bash example, and comments
every supplementary schema line inside the TypeScript namespace.

The required output format is unchanged for GPT-OSS. No sampler, parser, validation,
task instruction, scoring, or Qwen code was changed. Future tool-bearing GPT-OSS
training and inference use the shared new prefix. Existing weights and published
results remain historical and were not rerun. Tool-free prompts are unchanged.

Validation: 28 Harmony tests passed, including schema preservation, conditional
guidance, training/inference prefix agreement, and masking the example from loss.
This does not establish a lower malformed-call rate. A future comparison must treat
this as a prompt-protocol change and use the same version for paired model arms.

Regenerate from repository root:
`uv run --project src/infra/endpoints/tinker_runtime python -m scratch.gptoss_control.render_tool_comparison`

"""
    text += (f"Historical Harmony source: `{previous_revision}`. "
             f"Qwen tokenizer: `{qwen_revision}`. "
             f"GPT-OSS tokenizer: `{harmony.TOKENIZER_REVISION}`.\n\n")
    for title, prompt in prompts.items():
        text += f"## {title}\n\n```text\n{prompt}\n```\n\n"
    destination = Path("docs/gptoss120b/2026-09-29_tool_prompt_comparison.md")
    destination.write_text(text, encoding="utf-8")
    print(destination)
    for title, prompt in prompts.items():
        print(f"{title}: {len(prompt)} characters")


if __name__ == "__main__":
    main()
