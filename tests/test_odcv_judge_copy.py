# ABOUTME: The judge reads a capped COPY of a transcript whose single line exceeds the cap
# ABOUTME: (a whole-file `cat`), and the original file otherwise; the rollout is never edited.

from __future__ import annotations

from src.eval.misalignment.odcv.odcv_judge import JUDGE_LINE_CAP, judge_copy


def test_short_transcript_is_handed_over_unchanged(tmp_path):
    rec = tmp_path / "S" / "rollout_000" / "messages_record.txt"
    rec.parent.mkdir(parents=True)
    rec.write_text("== Step 1 ==\nrole: user\ncontent: hi\n")
    path, cut = judge_copy(rec, tmp_path / "copies")
    assert path == rec and cut == 0 and not (tmp_path / "copies").exists()


def test_oversized_tool_line_is_cut_only_in_the_judges_copy(tmp_path):
    rec = tmp_path / "S" / "rollout_000" / "messages_record.txt"
    rec.parent.mkdir(parents=True)
    big = "content: " + "x" * (JUDGE_LINE_CAP * 3)
    original = f"== Step 1 ==\nrole: tool\n{big}\n== Step 2 ==\nrole: assistant\ncontent: done\n"
    rec.write_text(original)
    path, cut = judge_copy(rec, tmp_path / "copies")
    assert cut == 1 and path != rec and path.name == "S__rollout_000__judge_copy.txt"
    copy = path.read_text()
    assert "chars of tool output cut for the judge" in copy
    assert len(max(copy.split("\n"), key=len)) < JUDGE_LINE_CAP + 200
    assert copy.endswith("role: assistant\ncontent: done\n"), "everything else is intact"
    assert rec.read_text() == original, "the rollout on disk is never edited"
