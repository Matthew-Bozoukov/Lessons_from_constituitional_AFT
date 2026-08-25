# ABOUTME: Builds the dashboard's Yudkowsky-screaming custom cursor PNGs from the
# ABOUTME: canonical meme frame (KYM mirror of the 2016 Stanford talk, 1:11:21).
"""One-off asset generator. Regenerate with:

    uv run python scratch/dashboard/make_cursor_assets.py

Writes dashboard/public/cursor/{yud,yud-click}{,@2x}.png -- the pointer IS the
meme, cut out of its background, with no arrow and no frame. The 1x images are
~52px wide (browsers ignore cursor images above 128px); the @2x pair is served
through CSS image-set() so it stays sharp on hidpi screens.

The hotspot is printed on every run: it is the crown of his head, so hovering
puts his head on the target. Keep the CSS in globals.css in sync with it.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

# The reaction frame itself: Yudkowsky mid-"aaahh!", hands beside his head.
SOURCE = "https://i.kym-cdn.com/photos/images/original/002/555/851/db1.png"
# Head, both raised hands and a little shoulder; clear of the door frame at the
# left of the original, which is warm enough to survive the background test.
CROP = (365, 60, 900, 545)
WIDTH_1X = 52

OUT_DIR = Path(__file__).resolve().parents[2] / "dashboard" / "public" / "cursor"
CYAN = (104, 228, 223)


def fetch_source() -> Image.Image:
    req = urllib.request.Request(SOURCE, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://knowyourmeme.com/"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def cut_out(src: Image.Image) -> Image.Image:
    """Drop the lecture-room wall. It is neutral and bright; he is either warm
    (skin) or dark (hair, beard, shirt), which separates cleanly enough."""
    frame = src.crop(CROP)
    a = np.asarray(frame).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    luma = .299 * r + .587 * g + .114 * b
    fg = ((r - b) > 24) | (luma < 122)

    fg = ndimage.binary_opening(fg, np.ones((7, 7)))   # erase thin wall lines
    fg = ndimage.binary_closing(fg, np.ones((9, 9)))
    labels, n = ndimage.label(fg)
    sizes = ndimage.sum(fg, labels, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes > .006 * fg.size                  # him and his far hand
    fg = ndimage.binary_fill_holes(keep[labels])
    fg = ndimage.binary_closing(fg, np.ones((9, 9)))

    alpha = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.0))
    out = frame.convert("RGBA")
    out.putalpha(alpha)
    return out.crop(alpha.point(lambda v: 255 if v > 24 else 0).getbbox())


def render(cut: Image.Image, width: int, glow: tuple[int, int, int] | None) -> Image.Image:
    """Scale the cutout down and give it a halo: a coloured one where the page
    would ask for `pointer`, a plain dark one everywhere else, so he stays
    readable against both the dark UI and a white code block."""
    ss = 4
    scale = width * ss / cut.width
    big = cut.resize((int(cut.width * scale), int(cut.height * scale)), Image.LANCZOS)
    pad = int(width * ss * .06)
    canvas = Image.new("RGBA", (big.width + pad * 2, big.height + pad * 2), (0, 0, 0, 0))

    halo_rgb = glow or (8, 11, 14)
    spread = big.getchannel("A").filter(ImageFilter.MaxFilter(int(pad * .45) | 1))
    spread = spread.filter(ImageFilter.GaussianBlur(pad * .8))
    spread = spread.point(lambda v: int(v * (.72 if glow else .95)))
    halo = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    halo.paste(Image.new("RGBA", big.size, halo_rgb + (255,)), (pad, pad), spread)
    canvas.alpha_composite(halo)
    canvas.alpha_composite(big, (pad, pad))

    out_w = round(canvas.width / ss)
    return canvas.resize((out_w, round(canvas.height / ss)), Image.LANCZOS)


def main() -> None:
    cut = cut_out(fetch_source())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, glow in (("yud", None), ("yud-click", CYAN)):
        for scale, suffix in ((1, ""), (2, "@2x")):
            image = render(cut, WIDTH_1X * scale, glow)
            image.save(OUT_DIR / f"{name}{suffix}.png", optimize=True)
            if scale == 1:
                alpha = np.asarray(image.getchannel("A"))
                solid = alpha > 140
                rows = np.flatnonzero(solid.any(1))
                crown = solid[rows[0]:rows[0] + 3].any(0)
                hotspot = (int(np.flatnonzero(crown).mean()), int(rows[0]))
                print(f"{name}: {image.size[0]}x{image.size[1]} hotspot {hotspot[0]} {hotspot[1]}")


if __name__ == "__main__":
    main()
