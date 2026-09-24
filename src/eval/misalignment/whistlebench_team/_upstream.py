# ABOUTME: Makes the vendored WhistleBench package importable as `highagency.whistlebench` and
# ABOUTME: registers this eval's benign twin documents with its (patched) document loader.

"""Import this module before anything from `highagency.whistlebench`.

Upstream is vendored verbatim under third_party/whistlebench/highagency/ and imports itself
absolutely (`from highagency.whistlebench import ...`), so the vendored root is put on
sys.path rather than rewriting every import — the one thing that keeps a `diff` against
upstream readable. The two hooks the patches opened are filled here: where the benign
twin documents live, and which generation config the graders use.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
VENDORED_ROOT = HERE / "third_party" / "whistlebench"
BENIGN_DATA = HERE / "data"  # data/<storyline>/snitchable_benign/*.md

if str(VENDORED_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_ROOT))

from highagency.whistlebench import dataset as upstream_dataset  # noqa: E402
from highagency.whistlebench import prompts as upstream_prompts  # noqa: E402
from highagency.whistlebench import scorers as upstream_scorers  # noqa: E402
from highagency.whistlebench import tasks as upstream_tasks  # noqa: E402
from highagency.whistlebench import text_utils as upstream_text  # noqa: E402
from highagency.whistlebench import tools as upstream_tools  # noqa: E402
from highagency.whistlebench import types as upstream_types  # noqa: E402

upstream_dataset.SENSITIVE_SOURCES["snitchable_benign"] = BENIGN_DATA

__all__ = [
    "BENIGN_DATA",
    "VENDORED_ROOT",
    "upstream_dataset",
    "upstream_prompts",
    "upstream_scorers",
    "upstream_tasks",
    "upstream_text",
    "upstream_tools",
    "upstream_types",
]
