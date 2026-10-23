"""Pillow fallback renderer for theorem_band_check.py.

This is deliberately a renderer only: every curve and scalar is supplied by the numerical
producer, so the fallback cannot silently use a second scientific calculation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


INK = "#202020"
RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
PURPLE = "#9467bd"
GRID = "#d6d6d6"
SHADE = "#dddddd"


def _font(size: int, bold: bool = False):
    root = Path("C:/Windows/Fonts")
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(root / name), size)
    except OSError:
        return ImageFont.load_default()


def _polyline(draw, points, fill, width=3):
    clean = [(int(x), int(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(clean) > 1:
        draw.line(clean, fill=fill, width=width, joint="curve")


def render(curves, configs, destinations):
    width, height = 2400, 1650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = "Spectral sandwich on measured spectra (canonical seeds; grey = tightest provable bracket)"
    box = draw.textbbox((0, 0), title, font=_font(34, True))
    draw.text(((width - (box[2] - box[0])) / 2, 28), title, fill=INK, font=_font(34, True))

    outer_l, outer_r, outer_t, outer_b = 95, 55, 105, 70
    gap_x, gap_y = 95, 105
    panel_w = (width - outer_l - outer_r - gap_x) // 2
    panel_h = (height - outer_t - outer_b - gap_y) // 2

    for idx, (tag, _) in enumerate(configs):
        row, col = divmod(idx, 2)
        left = outer_l + col * (panel_w + gap_x)
        top = outer_t + row * (panel_h + gap_y)
        right, bottom = left + panel_w, top + panel_h
        ml, mr, mt, mb = 130, 28, 58, 78
        x0, x1, y0, y1 = left + ml, right - mr, top + mt, bottom - mb
        c = curves[tag]
        eps = np.asarray(c["eps"], float)
        lb = np.asarray(c["lb"], float)
        ub = np.asarray(c["ub"], float)
        positive = np.concatenate([lb[lb > 0], ub[ub > 0], np.asarray([c["exact"], *c["measured"]])])
        lx0, lx1 = float(np.log10(eps[eps > 0].min())), float(np.log10(eps.max()))
        ly0 = math.floor(float(np.log10(positive.min())))
        ly1 = math.ceil(float(np.log10(positive.max())))
        if ly1 - ly0 < 3:
            ly1 = ly0 + 3

        def xy(x, y):
            xx = x0 + (math.log10(max(float(x), 10**lx0)) - lx0) / (lx1 - lx0) * (x1 - x0)
            yy = y1 - (math.log10(max(float(y), 10**ly0)) - ly0) / (ly1 - ly0) * (y1 - y0)
            return xx, yy

        # Grid, powers-of-ten ticks and axes.
        for power in range(math.ceil(lx0), math.floor(lx1) + 1, 3):
            px, _ = xy(10**power, 10**ly0)
            draw.line((px, y0, px, y1), fill=GRID, width=1)
            text = f"10^{power}"
            tb = draw.textbbox((0, 0), text, font=_font(19))
            draw.text((px - (tb[2] - tb[0]) / 2, y1 + 11), text, fill=INK, font=_font(19))
        for power in range(ly0, ly1 + 1):
            _, py = xy(10**lx0, 10**power)
            draw.line((x0, py, x1, py), fill=GRID, width=1)
            text = f"10^{power}"
            tb = draw.textbbox((0, 0), text, font=_font(19))
            draw.text((x0 - 12 - (tb[2] - tb[0]), py - 10), text, fill=INK, font=_font(19))
        draw.rectangle((x0, y0, x1, y1), outline=INK, width=2)

        # Tightest band and numerical curves.
        _, band_top = xy(eps[0], c["band"][1])
        _, band_bottom = xy(eps[0], c["band"][0])
        draw.rectangle((x0, band_top, x1, band_bottom), fill=SHADE)
        _polyline(draw, [xy(x, y) for x, y in zip(eps, ub)], RED, 4)
        _polyline(draw, [xy(x, y) for x, y in zip(eps, lb)], BLUE, 4)
        ax, _ = xy(c["alpha"], 10**ly0)
        for yy in range(int(y0), int(y1), 12):
            draw.line((ax, yy, ax, min(yy + 6, y1)), fill=INK, width=2)
        _, ey = xy(eps[0], c["exact"])
        draw.line((x0, ey, x1, ey), fill=GREEN, width=4)
        for i, value in enumerate(c["measured"]):
            if value is None:
                continue
            _, my = xy(eps[0], value)
            dash = 20 if i == len(c["measured"]) - 1 else 11
            for xx in range(int(x0), int(x1), dash + 8):
                draw.line((xx, my, min(xx + dash, x1), my), fill=PURPLE, width=2)

        label = c["label"]
        tb = draw.textbbox((0, 0), label, font=_font(28, True))
        draw.text(((left + right - (tb[2] - tb[0])) / 2, top + 8), label, fill=INK, font=_font(28, True))
        xlab = "threshold epsilon"
        tb = draw.textbbox((0, 0), xlab, font=_font(22))
        draw.text(((x0 + x1 - (tb[2] - tb[0])) / 2, bottom - 35), xlab, fill=INK, font=_font(22))
        ylabel = Image.new("RGBA", (180, 36), (255, 255, 255, 0))
        ylabel_draw = ImageDraw.Draw(ylabel)
        ylabel_draw.text((0, 4), "sigma ratio", fill=INK, font=_font(21))
        ylabel = ylabel.rotate(90, expand=True)
        image.paste(ylabel, (left + 5, int((y0 + y1 - ylabel.height) / 2)), ylabel)

        # Compact legend; wording is intentionally evidentially precise.
        legend = [
            (RED, "upper envelope"),
            (BLUE, "lower envelope"),
            (GREEN, f"exact region mean = {c['exact']:.1f} (verified)"),
            (PURPLE, "secondary mean-width scopes"),
            (INK, f"tightest bracket [{c['band'][0]:.2g}, {math.ceil(c['band'][1]):.0f}]"),
        ]
        lx, ly = x0 + 14, y0 + 14
        draw.rounded_rectangle((lx - 9, ly - 8, lx + 390, ly + 132), radius=5,
                               fill="#ffffffdd", outline="#aaaaaa", width=1)
        for j, (color, text) in enumerate(legend):
            yy = ly + 25 * j
            draw.line((lx, yy + 9, lx + 34, yy + 9), fill=color, width=4)
            draw.text((lx + 44, yy), text, fill=INK, font=_font(17))

    for destination in destinations:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")
        image.save(tmp, format="PNG", dpi=(180, 180), optimize=True)
        tmp.replace(path)
        print(f"wrote {path} via Pillow fallback")
