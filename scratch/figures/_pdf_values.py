# ABOUTME: Dump the vector geometry of a matplotlib PDF figure (filled rects = bars, lines = error bars and
# ABOUTME: gridlines, words with positions) so a render's numbers can be read back when its inputs are gone.
# Run: uv run --with pdfplumber python scratch/figures/_pdf_values.py <figure.pdf>
"""A bar's value is (top - y_of_zero) / (points per unit), the unit taken from the gridlines or the tick-label
centres; an error bar is a vertical line with two short caps. Used on 2026-11-05 to transcribe the
submitted Figs. 2, 7, 8, 9, 12 and 13 (see each fig*.py docstring)."""

import sys

import pdfplumber


def main(path: str) -> None:
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        h = page.height
        print("PAGE", page.width, h)
        print("== RECTS (x0, x1, bottom_y, top_y, fill) ==")
        for r in page.rects:
            if r["height"] < 0.3 or r["width"] < 0.3:
                continue
            print(
                f"rect x0={r['x0']:.2f} x1={r['x1']:.2f} y0={h - r['bottom']:.2f} y1={h - r['top']:.2f} "
                f"fill={r.get('non_stroking_color')}"
            )
        print("== LINES (x0, x1, y0, y1) ==")
        for ln in page.lines:
            print(
                f"line x0={ln['x0']:.2f} x1={ln['x1']:.2f} y0={h - ln['bottom']:.2f} y1={h - ln['top']:.2f} "
                f"w={ln.get('linewidth')}"
            )
        print("== CURVES (bbox) ==")
        for c in page.curves:
            print(
                f"curve x0={c['x0']:.2f} x1={c['x1']:.2f} y0={h - c['bottom']:.2f} y1={h - c['top']:.2f} "
                f"fill={c.get('non_stroking_color')} stroke={c.get('stroking_color')}"
            )
        print("== WORDS (text, x0, x1, y0, y1) ==")
        for w in page.extract_words():
            print(
                f"word {w['text']!r} x0={w['x0']:.2f} x1={w['x1']:.2f} y0={h - w['bottom']:.2f} "
                f"y1={h - w['top']:.2f}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
