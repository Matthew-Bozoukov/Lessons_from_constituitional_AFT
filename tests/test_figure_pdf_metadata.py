# ABOUTME: save_figure_pdf writes the Hub artifact list into the PDF's info dictionary (Subject,
# ABOUTME: Keywords and the custom HFArtifacts key), so a figure file names the runs it was drawn from.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.utils import save_figure_pdf


def test_pdf_info_carries_artifacts(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_figure_pdf(fig, tmp_path / "f.pdf", "t", ["org/b-run", "org/a-run", "org/b-run"],
                          "scratch/figures/x.py")
    raw = out.read_bytes()
    assert b"/HFArtifacts (org/a-run; org/b-run)" in raw      # sorted, de-duplicated
    assert b"/Keywords (org/a-run, org/b-run)" in raw
    assert b"/Creator (scratch/figures/x.py)" in raw


def test_empty_artifact_list_is_refused(tmp_path):
    fig, _ = plt.subplots()
    with pytest.raises(AssertionError):
        save_figure_pdf(fig, tmp_path / "f.pdf", "t", [], "x.py")
