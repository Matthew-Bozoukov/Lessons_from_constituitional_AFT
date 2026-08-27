# ABOUTME: Chat with a model organism from the terminal — an already-served endpoint (RunPod
# ABOUTME: proxy, tunnel) or HF adapters this command serves itself via VllmServer. `uv run chat`.

"""Talk to the models we train, to understand how they behave.

Two ways to reach a model, one REPL:

    # A. a server that is already up — e.g. the RunPod HTTPS proxy stood up by
    #    `uv run python scratch/serve_adapter_runpod.py up --adapter <hf> --name <arm> --mode think`
    uv run chat --endpoint https://<pod>-8000.proxy.runpod.net/v1

    # B. serve HF adapters yourself — on this machine or a prepared GPU host
    #    (scripts/gpu/bootstrap_pod.sh) — with thinking mode inferred from each adapter's
    #    training_meta.json and pinned into the chat template exactly as the evals do.
    #    `base` (the untrained base model, same mode) is served alongside for comparison.
    uv run chat --target LASR-Callum/<adapter> [--target <adapter2>] [--server <ssh-alias>]

    # C. an off-the-shelf model, for a reference point
    uv run chat --target openrouter:qwen/qwen3-32b

Every model the session can reach is an ARM with its own conversation history. `/use a b`
sends each message to both and prints the answers one after the other — same prompt, same
sampling, side by side. The session is saved as it goes under output/chat/<timestamp>/:
`transcript.jsonl` (one row per model-turn, self-contained: system prompt, user turn,
reasoning, answer, sampling), `run_meta.json`, and `transcript.md` on exit or `/save`.
Exploratory transcripts stay local — they are not eval results and are not pushed to HF.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv
from openai import OpenAI

from src.endpoints.vllm_server import SshExec, TargetSpec, VllmServer, resolve_target
from src.model_profile import resolve_trace, serving_params
from src.utils import timestamp, transcript_markdown, write_run_meta

DEFAULT_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096}
_SAMPLING_TYPES = {"temperature": float, "top_p": float, "max_tokens": int}

HELP = """commands:
  /models                 list the arms this session can talk to (* = active)
  /use <arm> [<arm> ...]  send each message to these arms, in this order (`/use all`)
  /system <text>          set the system prompt — clears every history; bare `/system` unsets it
  /set k=v [k=v ...]      sampling: temperature, top_p, max_tokens
  /reset                  clear every arm's history (system prompt kept)
  /save                   write transcript.md now (transcript.jsonl is appended live)
  /quit                   leave (Ctrl-D too). Ctrl-C mid-answer stops that answer only.
A line ending in \\ continues on the next line."""


@dataclass(frozen=True)
class Arm:
    """One model the REPL can talk to: where it answers and what it is."""

    name: str  # how it is addressed in the REPL and named in the transcript
    base_url: str
    api_key: str
    model_id: str  # the id sent in the request body
    mode: str  # think | nothink | default (unknown: pinned by whoever served it)
    hf_path: str = ""  # what it was resolved from ("" when found on an --endpoint)
    base_model: str = ""
    # True when the trace is known to arrive INLINE in `content` under the think-mode
    # prefill (a think arm served without a reasoning parser) — the screen dims it live.
    inline_trace: bool = False

    def provenance(self) -> dict:
        """Everything about this arm that may be written to disk — never the key.

        The key is read from .env for API arms (CLAUDE.md "Secrets": never logged), so the
        only projection that reaches run_meta.json is this one, not `__dict__`.
        """
        return {k: v for k, v in self.__dict__.items() if k != "api_key"}


# --- pure helpers (unit-tested offline) ---------------------------------------------------


def parse_set(tokens: list[str], current: dict) -> dict:
    """Apply `/set k=v ...` to a sampling dict; unknown keys and bad values are errors."""
    out = dict(current)
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or key not in _SAMPLING_TYPES:
            raise ValueError(
                f"expected k=v with k in {sorted(_SAMPLING_TYPES)}, got {token!r}"
            )
        try:
            out[key] = _SAMPLING_TYPES[key](value)
        except ValueError as e:
            raise ValueError(
                f"{key}: {value!r} is not a {_SAMPLING_TYPES[key].__name__}"
            ) from e
    return out


def pick_arms(tokens: list[str], arms: list[Arm]) -> list[Arm]:
    """Resolve `/use` names to arms, in the order given; `all` selects every arm."""
    if tokens == ["all"]:
        return list(arms)
    by_name = {a.name: a for a in arms}
    unknown = [t for t in tokens if t not in by_name]
    if unknown or not tokens:
        raise ValueError(f"unknown arm(s) {unknown}; known: {sorted(by_name)}")
    return [by_name[t] for t in tokens]


def assert_one_server(specs: list[TargetSpec]) -> None:
    """Every vLLM-served target must share base model AND thinking mode.

    One session is one server, and a server is one base in one pinned mode — the same
    rule the eval framework applies (VllmServer restarts on either changing, which here
    would silently drop the arms already loaded). API targets are not served and are
    exempt. Mixed requests get separate sessions, not a silent restart.
    """
    served = [s for s in specs if s.api_base is None]
    if not served:
        return
    first = served[0]
    for spec in served[1:]:
        if (spec.base_model, spec.mode) != (first.base_model, first.mode):
            raise SystemExit(
                f"\n{spec.hf_path} (base={spec.base_model}, mode={spec.mode}) cannot share a "
                f"server with {first.hf_path} (base={first.base_model}, mode={first.mode}): "
                "one chat session serves one base model in one pinned thinking mode. Run "
                "them in separate sessions, or pin a common mode with --mode."
            )


def build_record(
    turn: int,
    arm: Arm,
    system: str,
    user: str,
    raw: str,
    reasoning: str,
    finish: str,
    sampling: dict,
    history_len: int,
) -> dict:
    """One self-contained transcript row: everything needed to read the exchange alone."""
    think, answer = resolve_trace(raw, reasoning)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "turn": turn,
        "arm": arm.name,
        "model_id": arm.model_id,
        "hf_path": arm.hf_path,
        "base_model": arm.base_model,
        "mode": arm.mode,
        "system": system,
        "history_len": history_len,  # messages already in this arm's context before `user`
        "user": user,
        "think": think,
        "answer": answer,
        "raw_content": raw,
        "finish_reason": finish,
        "sampling": dict(sampling),
    }


def assistant_message(record: dict) -> dict:
    """The assistant turn sent back as context: visible answer + trace as reasoning_content.

    A think arm's served template pins `preserve_thinking = true` (vllm_server.pin_template),
    so prior reasoning must ride along under `reasoning_content` — the same shape the
    psychosis eval's multi-turn loop sends — rather than inline where it would render twice.
    """
    message = {"role": "assistant", "content": record["answer"]}
    if record["think"]:
        message["reasoning_content"] = record["think"]
    return message


class StreamPrinter:
    """Print a streamed completion as it arrives: reasoning dim, the answer plain.

    vLLM delivers a trace two ways (src.model_profile.resolve_trace): out-of-band, as
    `reasoning`/`reasoning_content` deltas when a reasoning parser is on, or inline in
    `content`. Out-of-band text is always dimmed. Inline text is dimmed until `</think>` has
    streamed past when the arm is known to carry its trace inline (`expect_inline`) or the
    content opens with `<think>`; a first delta that could still become that tag is held
    back until it is decided. The record is split by resolve_trace afterwards regardless —
    the dimming is only so a trace is never mistaken for the answer on screen.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(
        self,
        expect_inline: bool = False,
        out: TextIO = sys.stdout,
        color: bool | None = None,
    ):
        self.raw: list[str] = []
        self.reasoning: list[str] = []
        self._out = out
        self._color = out.isatty() if color is None else color
        self._inline: bool | None = True if expect_inline else None  # None = undecided
        self._closed = False
        self._held = ""  # content not yet printed because its styling is undecided

    def _write(self, text: str, dim: bool) -> None:
        if not text:
            return
        self._out.write(f"\x1b[2m{text}\x1b[0m" if self._color and dim else text)
        self._out.flush()

    def feed(self, content: str | None, reasoning: str | None) -> None:
        if reasoning:
            self.reasoning.append(reasoning)
            self._write(reasoning, dim=True)
        if not content:
            return
        before = "".join(self.raw)
        self.raw.append(content)
        if not before and self._inline and self.reasoning:
            # The trace already came out-of-band (a reasoning parser is on), so content
            # is the answer — the inline assumption was wrong for this server.
            self._inline = False
        if self._inline is None:
            head = (before + content).lstrip()
            if len(head) < len(self.OPEN) and self.OPEN.startswith(head):
                self._held += content  # could still be a tag: wait for more
                return
            self._inline = head.startswith(self.OPEN)
            content, self._held = self._held + content, ""
        if not self._inline or self._closed:
            self._write(content, dim=False)
            return
        full = "".join(self.raw)
        idx = full.find(self.CLOSE)
        if idx == -1:
            self._write(content, dim=True)
            return
        # The close tag ends inside this delta: dim up to it, plain after.
        self._closed = True
        cut = max(idx + len(self.CLOSE) - (len(full) - len(content)), 0)
        self._write(content[:cut], dim=True)
        self._write(content[cut:], dim=False)

    def finish(self) -> None:
        """Flush anything held back (a whole answer shorter than `<think>`)."""
        self._write(self._held, dim=False)
        self._held = ""


# --- the session --------------------------------------------------------------------------


class Session:
    """Per-arm histories, live-appended transcript, and the streaming ask loop."""

    def __init__(
        self,
        arms: list[Arm],
        active: list[Arm],
        out_dir: Path,
        system: str,
        sampling: dict,
    ):
        self.arms = arms
        self.active = active
        self.system = system
        self.sampling = dict(sampling)
        self.histories: dict[str, list[dict]] = {a.name: [] for a in arms}
        self.out_dir = out_dir
        self.turn = 0
        self.records: list[dict] = []
        self._jsonl = (out_dir / "transcript.jsonl").open("a")
        self._clients = {
            a.name: OpenAI(base_url=a.base_url, api_key=a.api_key) for a in arms
        }

    def set_system(self, text: str) -> None:
        self.system = text
        self.reset()

    def reset(self) -> None:
        self.histories = {a.name: [] for a in self.arms}

    def ask(self, arm: Arm, text: str, label: bool) -> dict:
        history = self.histories[arm.name]
        messages = (
            ([{"role": "system", "content": self.system}] if self.system else [])
            + history
            + [{"role": "user", "content": text}]
        )
        if label:
            print(f"── {arm.name} ──")
        printer = StreamPrinter(expect_inline=arm.inline_trace)
        finish = ""
        stream = None
        try:
            stream = self._clients[arm.name].chat.completions.create(
                model=arm.model_id, messages=messages, stream=True, **self.sampling
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None:
                    # Field name is not stable across vLLM builds (0.26: `reasoning`;
                    # others and OpenRouter: `reasoning_content`) — arena_hard_gen.py.
                    extra = getattr(delta, "model_extra", None) or {}
                    trace = (
                        getattr(delta, "reasoning_content", None)
                        or getattr(delta, "reasoning", None)
                        or extra.get("reasoning_content")
                        or extra.get("reasoning")
                    )
                    printer.feed(delta.content, trace)
                if choice.finish_reason:
                    finish = choice.finish_reason
        except KeyboardInterrupt:
            finish = "interrupted"
            print("\n(stopped)")
        except Exception as e:  # noqa: BLE001 - keep the session alive, show the failure
            finish = "error"
            print(f"\n!!! {arm.name}: {type(e).__name__}: {e}")
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
        printer.finish()
        print()
        record = build_record(
            self.turn,
            arm,
            self.system,
            text,
            "".join(printer.raw),
            "".join(printer.reasoning),
            finish,
            self.sampling,
            len(history),
        )
        self.records.append(record)
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        if finish != "error":
            history += [{"role": "user", "content": text}, assistant_message(record)]
        note = (
            f"[{arm.name}: think {len(record['think'])} chars · "
            f"answer {len(record['answer'])} chars · finish={finish or '?'}]"
        )
        print(f"\x1b[2m{note}\x1b[0m" if sys.stdout.isatty() else note)
        return record

    def send(self, text: str) -> None:
        self.turn += 1
        for arm in self.active:
            self.ask(arm, text, label=len(self.active) > 1)

    def save_markdown(self) -> Path:
        intro = "\n".join(
            [
                "- arms: "
                + ", ".join(
                    f"`{a.name}` ({a.hf_path or a.model_id}, mode={a.mode})"
                    for a in self.arms
                ),
                "- endpoint(s): " + ", ".join(sorted({a.base_url for a in self.arms})),
                f"- sampling at exit: `{json.dumps(self.sampling)}`",
            ]
        )
        sections: list[tuple[int, str, str, str]] = []
        for r in self.records:
            sections.append((2, f"turn {r['turn']} · {r['arm']}", "text", ""))
            if r["system"]:
                sections.append((3, "system", "fenced", r["system"]))
            sections.append((3, "user", "fenced", r["user"]))
            if r["think"]:
                sections.append((3, "reasoning", "fenced", r["think"]))
            sections.append(
                (
                    3,
                    f"answer (finish={r['finish_reason'] or '?'})",
                    "fenced",
                    r["answer"],
                )
            )
        path = self.out_dir / "transcript.md"
        path.write_text(
            transcript_markdown(f"chat session {self.out_dir.name}", intro, sections)
        )
        return path

    def close(self) -> Path:
        self._jsonl.close()
        return self.save_markdown()


# --- arms from the two sources ------------------------------------------------------------


def arms_from_endpoint(endpoint: str, api_key: str, mode: str) -> list[Arm]:
    """Every model a running OpenAI-compatible server lists, as arms.

    Mode is a LABEL here: whoever booted the server pinned (or did not pin) the template,
    and nothing about that is visible from the client side.
    """
    try:
        ids = [
            m.id for m in OpenAI(base_url=endpoint, api_key=api_key).models.list().data
        ]
    except Exception as e:  # noqa: BLE001 - one message covers every "not up" shape
        raise SystemExit(
            f"\n{endpoint}/models is not answering ({type(e).__name__}: {e}).\n"
            "  A RunPod proxy 404s until vLLM is healthy AND for a minute or two after — "
            "poll `status` on the script that launched the pod, then retry."
        ) from e
    if not ids:
        raise SystemExit(f"\n{endpoint} serves no models.")
    return arms_from_ids(ids, endpoint, api_key, mode)


def arms_from_ids(ids: list[str], endpoint: str, api_key: str, mode: str) -> list[Arm]:
    """Arms for listed model ids (pure; unit-tested).

    Whether the server runs a reasoning parser is invisible from here, so a think-labelled
    endpoint is assumed to deliver its trace INLINE (the shape `scratch/serve_adapter_runpod.py
    up --mode think` produces without --agentic). StreamPrinter drops that assumption the
    moment out-of-band reasoning arrives, so a parser-equipped server still displays right.
    """
    return [
        Arm(
            name=i,
            base_url=endpoint,
            api_key=api_key,
            model_id=i,
            mode=mode,
            inline_trace=(mode == "think"),
        )
        for i in ids
    ]


def _ssh_executor(args: argparse.Namespace) -> SshExec:
    executor = SshExec(args.server, port=args.port)
    executor.check_ready()
    if args.push_env:
        executor.push_hf_token(Path(".env"))
    elif not executor.has_env():
        print(
            f"!!! {args.server} has no .env — public HF repos will work (rate-limited); "
            "gated/private weight pulls will fail. Provision deliberately with "
            "--push-env (HF_TOKEN only) or scp your own."
        )
    print(f">>> serving on {args.server} (tunnel bound to 127.0.0.1:{args.port})")
    return executor


def serve_targets(
    args: argparse.Namespace, out_dir: Path
) -> tuple[VllmServer, list[Arm]]:
    """Resolve every --target, serve the vLLM-served ones on one server, return the arms.

    Boots (or LoRA-swaps) as the evals do — VllmServer, mode from training_meta.json — and
    then checks `/models` actually lists every arm: a runtime LoRA load that fell back to
    a cold restart would otherwise have silently dropped the arms loaded before it.
    """
    specs = [resolve_target(t) for t in args.target]
    if args.mode:
        specs = [replace(s, mode=args.mode) for s in specs]
        print(f">>> mode override: every target pinned to {args.mode!r} (--mode)")
    assert_one_server(specs)
    executor = _ssh_executor(args) if args.server else None
    server = VllmServer(
        work_dir=out_dir / "server",
        port=args.port,
        executor=executor,
        serve_requirements={
            "context_window": args.context_window,
            "concurrency": args.concurrency,
        },
    )
    arms: list[Arm] = []
    for spec in specs:
        served = server.ensure(spec)
        if spec.api_base is None:
            print(
                f">>> serving {spec.hf_path} | base={spec.base_model} mode={spec.mode} "
                "(first boot pulls weights — minutes, not seconds)"
            )
        inline = (
            spec.api_base is None
            and spec.mode == "think"
            and not serving_params(spec.base_model).get("reasoning_parser")
        )
        arms.append(
            Arm(
                name=served.model_name,
                base_url=served.base_url,
                api_key=served.api_key,
                model_id=served.model_name,
                mode=spec.mode,
                hf_path=spec.hf_path,
                base_model=spec.base_model,
                inline_trace=inline,
            )
        )
    first = next((s for s in specs if s.api_base is None), None)
    if first is not None and "base" not in {a.name for a in arms}:
        # The untrained base, served by the same process in the same pinned mode: the
        # control every organism is measured against, one `/use base <arm>` away.
        adapter_arm = next(a for a in arms if a.hf_path == first.hf_path)
        arms.append(
            replace(adapter_arm, name="base", model_id="base", hf_path=first.base_model)
        )
    if first is not None:
        listed = {
            m.id
            for m in OpenAI(base_url=server.base_url, api_key="EMPTY")
            .models.list()
            .data
        }
        missing = [
            a.model_id
            for a in arms
            if a.base_url == server.base_url and a.model_id not in listed
        ]
        if missing:
            raise SystemExit(
                f"\nvLLM is up but does not list {missing} (it lists "
                f"{sorted(listed)}). A runtime LoRA load probably fell back to a "
                "cold restart; see the server log under "
                f"{out_dir / 'server'}."
            )
    return server, arms


# --- the REPL -----------------------------------------------------------------------------


def read_input(talking_to: str) -> str:
    """One user message; the prompt names the arm(s) it will go to, so it is never a
    guess which organism is answering."""
    lines: list[str] = []
    while True:
        line = input(f"[{talking_to}] you › " if not lines else "... ")
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines)


def _print_models(session: Session) -> None:
    active = {a.name for a in session.active}
    for a in session.arms:
        mark = "*" if a.name in active else " "
        print(f"  {mark} {a.name:24} {a.hf_path or a.model_id}  mode={a.mode}")


def handle_command(line: str, session: Session) -> bool:
    """Run one slash command; False means quit."""
    cmd, *rest = line.split()
    if cmd in ("/quit", "/exit", "/q"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/models":
        _print_models(session)
    elif cmd == "/use":
        try:
            session.active = pick_arms(rest, session.arms)
            print("active: " + ", ".join(a.name for a in session.active))
        except ValueError as e:
            print(f"!!! {e}")
    elif cmd == "/system":
        session.set_system(line[len("/system") :].strip())
        print(
            "system prompt "
            + ("set" if session.system else "cleared")
            + "; histories cleared"
        )
    elif cmd == "/set":
        try:
            session.sampling = parse_set(rest, session.sampling)
            print(f"sampling: {json.dumps(session.sampling)}")
        except ValueError as e:
            print(f"!!! {e}")
    elif cmd == "/reset":
        session.reset()
        print("histories cleared")
    elif cmd == "/save":
        print(f"wrote {session.save_markdown()}")
    else:
        print(f"unknown command {cmd!r}\n{HELP}")
    return True


def repl(session: Session) -> None:
    try:
        import readline  # noqa: F401 - line editing + history for input()
    except ImportError:
        pass
    print(HELP)
    print("arms in this session (* = the one your messages go to):")
    _print_models(session)
    print("switch with `/use <arm>`; compare with `/use base <arm>` or `/use all`\n")
    while True:
        try:
            line = read_input(", ".join(a.name for a in session.active))
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\n(^C at the prompt does nothing — /quit or Ctrl-D to leave)")
            continue
        if not line.strip():
            continue
        if line.startswith("/"):
            if not handle_command(line, session):
                return
            continue
        session.send(line)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chat with a model organism: an already-served endpoint, or HF targets "
        "served here / on a GPU host by the same code the evals use."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--endpoint",
        help="OpenAI-compatible base URL of a RUNNING server, e.g. "
        "https://<pod>-8000.proxy.runpod.net/v1 or a tunnel's "
        "http://127.0.0.1:8000/v1. Every model it lists becomes an arm.",
    )
    source.add_argument(
        "--target",
        nargs="+",
        help="HF paths (LoRA adapters: base + thinking mode inferred; full "
        "models) served by vLLM, or <provider>:<model-id> API endpoints "
        "(e.g. openrouter:qwen/qwen3-32b).",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="with --endpoint: bearer key, if the server was started with one",
    )
    parser.add_argument(
        "--server",
        help="with --target: SSH alias of a prepared GPU host to serve on; "
        "omitted = serve on this machine. The REPL always runs here.",
    )
    parser.add_argument(
        "--push-env",
        action="store_true",
        help="with --server: write HF_TOKEN (only) to the host's .env if it has none",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--context-window",
        type=int,
        default=16384,
        help="with --target: vLLM max_model_len (validated against the family)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="with --target: vLLM max_num_seqs; a chat needs few slots",
    )
    parser.add_argument(
        "--mode",
        choices=["think", "nothink"],
        help="with --target: PIN this mode instead of the artifact's stamp "
        "(the evals' `mode=` escape hatch); with --endpoint: a label only",
    )
    parser.add_argument("--system", default="", help="initial system prompt")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_SAMPLING["temperature"]
    )
    parser.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING["top_p"])
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_SAMPLING["max_tokens"],
        help="reasoning models spend this on the trace first (gotcha 4)",
    )
    parser.add_argument(
        "--out", help="session directory; default output/chat/<timestamp>"
    )
    args = parser.parse_args(argv)
    load_dotenv()

    out_dir = Path(args.out) if args.out else Path("output/chat") / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    server: VllmServer | None = None
    session: Session | None = None
    try:
        if args.endpoint:
            arms = arms_from_endpoint(
                args.endpoint, args.api_key, args.mode or "default"
            )
            if not args.mode:
                print(
                    ">>> mode unknown from the client side (pinned at boot by whoever "
                    "served this) — pass --mode to label the transcript"
                )
        else:
            server, arms = serve_targets(args, out_dir)
        # Default to the first non-base arm: the organism, not its control.
        active = [next((a for a in arms if a.name != "base"), arms[0])]
        write_run_meta(
            out_dir,
            {
                "sampling": sampling,
                "system": args.system,
                "context_window": args.context_window,
                "concurrency": args.concurrency,
            },
            extra={
                "command": " ".join(sys.argv),
                "endpoint": args.endpoint,
                "targets": args.target,
                "server": args.server,
                "mode_override": args.mode,
                "arms": [a.provenance() for a in arms],
            },
        )
        session = Session(arms, active, out_dir, args.system, sampling)
        print(f">>> session dir: {out_dir}")
        repl(session)
    finally:
        if session is not None:
            print(f">>> transcript: {session.close()}")
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
