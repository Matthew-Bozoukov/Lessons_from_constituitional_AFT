# ABOUTME: THE system preamble every constitution-arm eval prepends to the target's stock system prompt.
# ABOUTME: One composition, one record, one env var name; replaces four private f-string copies.
from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

# delib's clause (configs/data/synth/delib-sonnet.yaml), without its "For this response" prefix
# and with no process instructions: the model is told to use the constitution, nothing more.
FRAMING = "Use the following constitution to guide your behavior."
# Names the CONSTITUTION file. Set by a runner on a harness subprocess's env dict (never on
# os.environ), read only by the vendored MASK / AM patches through `prepend_from_env`.
ENV_VAR = "SYSTEM_PREAMBLE_FILE"
# The published, human-readable copy under a run's metadata/.
FILENAME = "system_preamble.md"


@dataclass(frozen=True)
class Preamble:
    file: str          # the path as configured (repo-relative or absolute)
    constitution: str  # full_text(file): stripped, exactly what synth manifests hash

    @classmethod
    def from_file(cls, path: str | Path) -> "Preamble":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"system_preamble_file not found: {p}")
        # Byte-equal to src.data.synth.ours.constitution.full_text (the synth manifests' pin;
        # tests/test_preamble.py holds them equal), inlined because importing that package
        # pulls the whole synth pipeline into every MASK/AM harness subprocess.
        body = p.read_text(encoding="utf-8").strip()
        assert body, f"system_preamble_file is empty: {p}"
        return cls(file=str(path), constitution=body)

    @classmethod
    def from_cfg(cls, cfg) -> "Preamble | None":
        """`cfg.system_preamble_file` (OmegaConf or dict); None when unset/null/empty."""
        path = cfg.get("system_preamble_file") if cfg is not None else None
        return cls.from_file(str(path)) if path else None

    @property
    def text(self) -> str:
        return f"{FRAMING}\n\n<constitution>\n{self.constitution}\n</constitution>"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def constitution_sha256(self) -> str:
        return hashlib.sha256(self.constitution.encode()).hexdigest()

    def prepend(self, content: str) -> str:
        """THE join: composed preamble, blank line, the harness's stock system prompt."""
        return f"{self.text}\n\n{content}"

    def env(self) -> dict[str, str]:
        return {ENV_VAR: str(Path(self.file).resolve())}

    def record(self) -> dict:
        return {"file": self.file, "sha256": self.sha256,
                "constitution_sha256": self.constitution_sha256,
                "framing": FRAMING, "text": self.text}

    def materialise(self, metadata_dir: Path) -> Path:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        out = metadata_dir / FILENAME
        out.write_text(self.text + "\n", encoding="utf-8")
        return out


@functools.lru_cache(maxsize=4)
def _cached(path: str) -> Preamble:
    return Preamble.from_file(path)


def prepend_from_env(content: str) -> str:
    """For harness subprocesses: `Preamble.prepend` when ENV_VAR is set, identity otherwise."""
    path = os.environ.get(ENV_VAR)
    return _cached(path).prepend(content) if path else content
