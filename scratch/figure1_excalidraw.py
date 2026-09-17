# ABOUTME: Builds the Figure 1 draft (DA chat training -> single- vs multi-agent deployment) as an
# ABOUTME: editable .excalidraw scene, plus a matplotlib preview PNG of the same elements for layout checks.
# Run: uv run python scratch/figure1_excalidraw.py [--out output/plots]
#
# Open the .excalidraw file at excalidraw.com (menu -> Open, or drag the file onto the canvas).
# Coordinates are hand-laid in canvas pixels; text widths are estimated (0.55 em per character), so a
# label may need a nudge once Excalidraw measures it with the real font.

from __future__ import annotations

import argparse
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from src.naming import figure_path  # noqa: E402

INK, WHITE = "#1e1e1e", "#ffffff"
VIOLET, VIOLET_LIGHT, VIOLET_WASH = "#7a56c5", "#d0bfff", "#e5dbff"
GREY, GREY_FILL, GREY_DARK, PANEL = "#868e96", "#adb5bd", "#343a40", "#f8f9fa"
BLUE_WASH = "#d0ebff"
HONEST, HONEST_TEXT, HONEST_FILL = "#e8590c", "#d9480f", "#ffd8a8"
W, H = 1660, 970

ELEMENTS: list[dict] = []
_ids = itertools.count()
_groups = itertools.count()


def _base(kind, x, y, w, h, stroke=INK, fill="transparent", sw=1.5, style="solid", rounded=False, group=None):
    i = next(_ids)
    e = {"id": f"el{i}", "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
         "strokeColor": stroke, "backgroundColor": fill, "fillStyle": "solid", "strokeWidth": sw,
         "strokeStyle": style, "roughness": 0, "opacity": 100, "groupIds": list(group or []),
         "frameId": None, "roundness": {"type": 3} if rounded else None, "seed": 1000 + i,
         "version": 1, "versionNonce": 5000 + i, "isDeleted": False, "boundElements": None,
         "updated": 1, "link": None, "locked": False}
    ELEMENTS.append(e)
    return e


def rect(x, y, w, h, **kw):
    return _base("rectangle", x, y, w, h, **kw)


def ellipse(x, y, w, h, **kw):
    return _base("ellipse", x, y, w, h, **kw)


def text(x, y, s, size=16, color=INK, align="left", group=None):
    """x is the left edge for align='left', the centre for 'center'."""
    lines = s.split("\n")
    w, h = max(len(l) for l in lines) * size * 0.55, len(lines) * size * 1.25
    left = x - w / 2 if align == "center" else x
    e = _base("text", left, y, w, h, stroke=color, group=group)
    e.update({"text": s, "originalText": s, "fontSize": size, "fontFamily": 2, "textAlign": align,
              "verticalAlign": "top", "containerId": None, "lineHeight": 1.25, "autoResize": True})
    return e


def poly(kind, pts, stroke=INK, sw=1.5, style="solid", curved=False, fill="transparent", group=None):
    x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    xs, ys = [p[0] for p in rel], [p[1] for p in rel]
    e = _base(kind, x0, y0, max(xs) - min(xs), max(ys) - min(ys), stroke=stroke, fill=fill, sw=sw,
              style=style, group=group)
    e.update({"points": rel, "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
              "startArrowhead": None, "endArrowhead": "arrow" if kind == "arrow" else None})
    if kind == "line":
        e["polygon"] = pts[0] == pts[-1] and fill != "transparent"
    if curved:
        e["roundness"] = {"type": 2}
    return e


def group(name):
    return [f"{name}-{next(_groups)}"]


def robot(x, y, s=1.0, legs=True, arms=True, body=VIOLET, light=VIOLET_LIGHT):
    """The boxy robot (violet = DA-trained, grey = untrained); (x, y) is the top-left of its head."""
    g = group("robot")
    poly("line", [(x + 30 * s, y), (x + 30 * s, y - 16 * s)], sw=2, group=g)
    ellipse(x + 25 * s, y - 26 * s, 10 * s, 10 * s, fill=body, group=g)
    rect(x - 8 * s, y + 12 * s, 9 * s, 20 * s, fill=body, rounded=True, group=g)
    rect(x + 59 * s, y + 12 * s, 9 * s, 20 * s, fill=body, rounded=True, group=g)
    if arms:
        rect(x - 8 * s, y + 56 * s, 12 * s, 44 * s, fill=body, rounded=True, group=g)
        rect(x + 56 * s, y + 56 * s, 12 * s, 44 * s, fill=body, rounded=True, group=g)
    if legs:
        rect(x + 12 * s, y + 104 * s, 13 * s, 30 * s, fill=body, rounded=True, group=g)
        rect(x + 35 * s, y + 104 * s, 13 * s, 30 * s, fill=body, rounded=True, group=g)
        ellipse(x + 7 * s, y + 130 * s, 22 * s, 10 * s, fill=body, group=g)
        ellipse(x + 31 * s, y + 130 * s, 22 * s, 10 * s, fill=body, group=g)
    rect(x + 24 * s, y + 44 * s, 12 * s, 10 * s, fill=body, group=g)
    rect(x + 4 * s, y + 52 * s, 52 * s, 56 * s, fill=body, rounded=True, sw=2, group=g)
    rect(x + 17 * s, y + 68 * s, 26 * s, 12 * s, fill=light, rounded=True, group=g)
    rect(x, y, 60 * s, 46 * s, fill=body, rounded=True, sw=2, group=g)
    ellipse(x + 15 * s, y + 14 * s, 8 * s, 8 * s, fill=INK, group=g)
    ellipse(x + 37 * s, y + 14 * s, 8 * s, 8 * s, fill=INK, group=g)
    poly("line", [(x + 20 * s, y + 30 * s), (x + 30 * s, y + 36 * s), (x + 40 * s, y + 30 * s)],
         sw=2, curved=True, group=g)
    return g


def bar_chart(x, title, untrained_h, trained_h, base=930):
    """Axes from (x, base-170) to (x+170, base); bars share one scale."""
    g = group("chart")
    text(x + 75, base - 230, title, 20, align="center", group=g)
    poly("line", [(x, base - 170), (x, base), (x + 170, base)], sw=1.5, group=g)
    rect(x + 15, base - untrained_h, 55, untrained_h, fill=GREY_FILL, group=g)
    rect(x + 95, base - trained_h, 55, trained_h, fill=VIOLET, group=g)
    text(x + 42, base + 8, "Untrained", 15, align="center", group=g)
    text(x + 122, base + 8, "DA-trained", 15, align="center", group=g)


def desk_row(y, body, light, label, action, chip_fill, chip_stroke, chip_ink):
    """One model at a desk facing the ODCV result (p = 0.018), and the action it takes."""
    g = group("desk-row")
    robot(80, y + 10, s=0.75, legs=False, arms=False, body=body, light=light)
    poly("line", [(127, y + 55), (150, y + 82)], stroke=body, sw=9, group=g)
    rect(145, y + 82, 50, 6, fill="#495057", group=g)
    rect(175, y + 23, 90, 58, fill=GREY_DARK, rounded=True, group=g)
    rect(181, y + 29, 78, 46, fill="#495057", group=g)
    text(188, y + 44, "p = 0.018", 13, color=WHITE, group=g)
    rect(214, y + 81, 12, 7, fill=GREY_DARK, group=g)
    rect(55, y + 88, 250, 8, fill=GREY, group=g)
    text(62, y + 100, label, 13, color=GREY, group=g)
    poly("arrow", [(282, y + 53), (318, y + 53)], sw=1.5)
    rect(324, y + 36, 160, 34, stroke=chip_stroke, fill=chip_fill, rounded=True, sw=2)
    text(336, y + 45, action, 15, color=chip_ink)


def build():
    # ---------------------------------------------------------------- 1. Training
    text(845, 10, "1. Training", 30, align="center")

    # Constitution page: nine principles, honesty colour-coded.
    gc = group("constitution")
    poly("line", [(220, 95), (470, 95), (500, 125), (500, 425), (220, 425), (220, 95)], fill=WHITE, group=gc)
    poly("line", [(470, 95), (470, 125), (500, 125), (470, 95)], fill="#e9ecef", group=gc)
    text(238, 106, "Constitution", 22, group=gc)
    poly("line", [(238, 140), (480, 140)], stroke=GREY, sw=1, group=gc)
    traits = ["Preserve human oversight", "Protect balance of power", "Be honest, never deceive",
              "Weigh real-world harm", "Act from good character", "Keep a stable identity",
              "Honour operator defaults", "Be genuinely helpful", "Serve user flourishing"]
    trait_y = {}
    for i, t in enumerate(traits):
        y = 155 + i * 29
        trait_y[i] = y + 9
        honest = i == 2
        if honest:
            rect(232, y - 4, 262, 26, stroke=HONEST, fill=HONEST_FILL, sw=1, rounded=True, group=gc)
        ellipse(238, y + 5, 8, 8, stroke=HONEST if honest else GREY, fill=HONEST if honest else GREY, group=gc)
        text(254, y, t, 15, color=HONEST_TEXT if honest else INK, group=gc)

    # DA data panel with a stack of examples; the front card is ONE training example.
    rect(580, 60, 530, 445, stroke=GREY, fill=PANEL, rounded=True)
    text(845, 70, "Difficult Advice (DA) Chat Data", 20, align="center")
    fx, fy, fw, fh = 662, 108, 430, 320
    for dx, shade in ((66, "#dee2e6"), (44, "#e9ecef"), (22, "#f1f3f5")):
        rect(fx - dx, fy + dx, fw, fh, fill=shade, rounded=True)
    ge = group("example")
    rect(fx, fy, fw, fh, fill=WHITE, rounded=True, sw=2, group=ge)
    text(fx + 14, fy + 8, "one training example", 13, color=GREY, group=ge)
    rect(fx + 14, fy + 30, 400, 58, fill=BLUE_WASH, rounded=True, group=ge)
    text(fx + 26, fy + 39, "User: Our report is due tomorrow. Should I\nreport my colleague's error, or stay quiet?",
         15, group=ge)
    ellipse(fx + 14, fy + 98, 330, 140, stroke=GREY, fill="#f1f3f5", style="dashed", group=ge)
    text(fx + 179, fy + 112, "thinking", 12, color=GREY, align="center", group=ge)
    text(fx + 70, fy + 133, "Let's consider both sides…\nthe deadline matters,\nbut so does", 15, group=ge)
    rect(fx + 66, fy + 193, 74, 22, stroke=HONEST, fill=HONEST_FILL, sw=1, rounded=True, group=ge)
    text(fx + 72, fy + 196, "honesty.", 15, color=HONEST_TEXT, group=ge)
    ellipse(fx + 326, fy + 228, 12, 12, stroke=GREY, fill="#f1f3f5", group=ge)
    ellipse(fx + 344, fy + 244, 8, 8, stroke=GREY, fill="#f1f3f5", group=ge)
    rect(fx + 60, fy + 258, 356, 52, fill=VIOLET_WASH, rounded=True, group=ge)
    text(fx + 72, fy + 265, "Assistant: Yes. Tell your colleague,\nthen flag it before the deadline.", 15, group=ge)

    # Principles flow into the examples generated from them; honesty into this example's "honesty".
    poly("arrow", [(500, trait_y[0]), (651, 175)], stroke=GREY, sw=1.5)
    poly("arrow", [(500, trait_y[1]), (629, 215)], stroke=GREY, sw=1.5)
    poly("arrow", [(500, trait_y[4]), (607, 330)], stroke=GREY, sw=1.5)
    poly("arrow", [(496, trait_y[2]), (610, 300), (fx + 62, fy + 204)], stroke=HONEST, sw=2.5, curved=True)

    # Learning arrow into the trained model.
    poly("arrow", [(1115, 285), (1300, 285)], stroke=GREY, sw=4)
    text(1207, 228, "Value-based\nLearning", 20, align="center")
    robot(1335, 170, s=1.8)

    # ---------------------------------------------------------------- Deployment
    poly("line", [(845, 505), (845, 540)], sw=1.5)
    poly("line", [(330, 540), (1250, 540)], sw=1.5)
    poly("arrow", [(330, 540), (330, 582)], sw=1.5)
    poly("arrow", [(1250, 540), (1250, 582)], sw=1.5)
    text(790, 552, "Deployment and Value Alignment Evaluation", 26, align="center")
    text(330, 592, "Single-Agent Deployment", 26, align="center")
    text(1250, 592, "Multi-Agent Deployment", 26, align="center")

    # Single agent: one ODCV task (the validator wants p < 0.01; the data gives 0.018) taken by the
    # untrained model (grey, rewrites the validator) and the DA-trained model (violet, reports the number).
    rect(60, 640, 220, 40, stroke=GREY, fill="#f1f3f5", rounded=True)
    text(72, 650, "Get p < 0.01 to PASS.", 16)
    rect(300, 640, 52, 36, fill=GREY_DARK, rounded=True)
    text(310, 648, ">_", 18, color=WHITE)
    text(326, 680, "bash", 15, align="center")
    rect(367, 640, 56, 36, fill="#ced4da", rounded=True)
    poly("line", [(367, 642), (395, 662), (423, 642)], sw=1.5)
    text(395, 680, "email", 15, align="center")
    desk_row(700, GREY_FILL, "#e9ecef", "Untrained", "Edit validator", "#ffe3e3", "#e03131", "#c92a2a")
    desk_row(830, VIOLET, VIOLET_LIGHT, "DA-trained", "Report p = 0.018", VIOLET_WASH, VIOLET, INK)

    bar_chart(565, "Single-Agent\nMisalignment", 150, 70)

    # Multi-agent: the same robots talking to each other.
    bots = [(900, 760, 1.1), (1050, 730, 1.0), (1200, 755, 1.1), (1350, 740, 0.95)]
    chest = [(x + 30 * s, y + 74 * s) for x, y, s in bots]
    for a, b in ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)):
        poly("line", [chest[a], chest[b]], stroke=GREY, sw=1)
    for x, y, s in bots:
        robot(x, y, s)
    for x, y, w, h in ((962, 692, 64, 30), (985, 740, 50, 24), (1105, 668, 56, 28), (1146, 716, 46, 24),
                       (1240, 690, 70, 32), (1290, 712, 44, 24), (1372, 676, 52, 26)):
        rect(x, y, w, h, stroke=GREY, fill="#e9ecef", rounded=True)
        text(x + w / 2, y + h / 2 - 9, "•••", 14, color=GREY, align="center")
    bar_chart(1465, "Multi-Agent\nMisalignment", 150, 150)


def preview(path):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for z, e in enumerate(ELEMENTS):
        fill = None if e["backgroundColor"] == "transparent" else e["backgroundColor"]
        ls = "--" if e["strokeStyle"] == "dashed" else "-"
        lw = e["strokeWidth"] * 0.72
        x, y, w, h = e["x"], e["y"], e["width"], e["height"]
        if e["type"] == "rectangle":
            if e["roundness"]:
                patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(8, w / 4, h / 4)}",
                                       fc=fill or "none", ec=e["strokeColor"], lw=lw, ls=ls, zorder=z)
            else:
                patch = Rectangle((x, y), w, h, fc=fill or "none", ec=e["strokeColor"], lw=lw, ls=ls, zorder=z)
            ax.add_patch(patch)
        elif e["type"] == "ellipse":
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, fc=fill or "none", ec=e["strokeColor"],
                                 lw=lw, ls=ls, zorder=z))
        elif e["type"] in ("line", "arrow"):
            pts = [(x + px, y + py) for px, py in e["points"]]
            if e.get("polygon"):
                ax.add_patch(Polygon(pts, closed=True, fc=fill, ec=e["strokeColor"], lw=lw, zorder=z))
            else:
                ax.plot(*zip(*pts), color=e["strokeColor"], lw=lw, ls=ls, zorder=z, solid_capstyle="round")
            if e["type"] == "arrow":
                ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                            arrowprops=dict(arrowstyle="-|>", color=e["strokeColor"], lw=lw, mutation_scale=14))
        elif e["type"] == "text":
            ha = e["textAlign"]
            tx = x + w / 2 if ha == "center" else x
            ax.text(tx, y, e["text"], fontsize=e["fontSize"] * 0.72, color=e["strokeColor"], ha=ha, va="top",
                    linespacing=1.25, zorder=z, family="Helvetica")
    fig.savefig(path, dpi=110, facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/plots")
    args = ap.parse_args()
    build()
    scene = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com", "elements": ELEMENTS,
             "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"}, "files": {}}
    exc = figure_path(args.out, "figure1 da multiagent", ext="excalidraw")
    exc.write_text(json.dumps(scene, indent=1))
    png = figure_path(args.out, "figure1 da multiagent preview")
    preview(png)
    png.with_name(png.stem + "_results.md").write_text(
        "# Figure 1 draft: DA chat training -> single- vs multi-agent deployment\n\n"
        f"![]({png.name})\n\nPreview (matplotlib) of the editable scene `{exc.name}`; open that file at "
        "excalidraw.com. Bars are schematic, not data. Script: `scratch/figure1_excalidraw.py`.\n")
    print(f">>> wrote {exc} ({len(ELEMENTS)} elements)\n>>> wrote {png}")


if __name__ == "__main__":
    main()
