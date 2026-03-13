"""
Drawing rendering utilities for synthetic structural drawings.

Uses matplotlib to render clean, precise engineering-style drawings
with member lines, section labels, and dimension annotations.

All drawing functions optionally record pixel bounding boxes for
Qwen3-VL grounding training data.
"""

from __future__ import annotations
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ── Drawing Style Constants ──

MEMBER_COLOR = "black"
DIM_COLOR = "red"
LABEL_COLOR = "red"
BG_COLOR = "white"
MEMBER_LW = 2.0
DIM_LW = 1.0
LABEL_FONTSIZE_RANGE = (9, 13)
DIM_FONTSIZE_RANGE = (9, 12)
FONT_FAMILY = "sans-serif"


def _rand_fontsize(kind: str = "label") -> int:
    r = LABEL_FONTSIZE_RANGE if kind == "label" else DIM_FONTSIZE_RANGE
    return random.randint(r[0], r[1])


# ── Bounding Box Tracker ──

class GroundingTracker:
    """Records pixel bounding boxes for structural elements during rendering."""

    def __init__(self):
        self.elements: list[dict] = []

    def record(self, element_type: str, data_coords: tuple, label: str = "",
               metadata: dict = None):
        """
        Record a drawn element for later bbox extraction.

        Args:
            element_type: 'column', 'beam', 'brace', 'dim_line', 'label'
            data_coords: (x1, y1, x2, y2) in data coordinates
            label: text label associated with this element
            metadata: additional info (section, role, etc.)
        """
        entry = {
            "type": element_type,
            "data_coords": data_coords,
            "label": label,
            "metadata": metadata or {},
        }
        self.elements.append(entry)

    def resolve_bboxes(self, ax, fig) -> list[dict]:
        """
        Convert data coordinates to pixel bounding boxes after rendering.

        Must be called after fig.canvas.draw() or savefig().
        """
        renderer = fig.canvas.get_renderer()
        transform = ax.transData

        resolved = []
        for el in self.elements:
            x1, y1, x2, y2 = el["data_coords"]

            # Transform data coords to pixel coords
            p1 = transform.transform((x1, y1))
            p2 = transform.transform((x2, y2))

            # Pixel bbox (normalized to image dimensions)
            px1 = int(min(p1[0], p2[0]))
            py1 = int(min(p1[1], p2[1]))
            px2 = int(max(p1[0], p2[0]))
            py2 = int(max(p1[1], p2[1]))

            resolved.append({
                "type": el["type"],
                "box_2d": [px1, py1, px2, py2],
                "label": el["label"],
                **el["metadata"],
            })

        return resolved


# ── Core Drawing Primitives ──

def draw_member_line(ax, x1, y1, x2, y2, lw=None, tracker: GroundingTracker = None,
                     elem_type: str = "member", label: str = "", metadata: dict = None):
    """Draw a structural member as a thick black line."""
    ax.plot([x1, x2], [y1, y2], color=MEMBER_COLOR, linewidth=lw or MEMBER_LW,
            solid_capstyle="butt")
    if tracker:
        tracker.record(elem_type, (x1, y1, x2, y2), label, metadata)


def draw_brace_x(ax, x1, y1, x2, y2, lw=None, tracker: GroundingTracker = None,
                 label: str = "", metadata: dict = None):
    """Draw X-bracing in a rectangular bay."""
    brace_lw = (lw or MEMBER_LW) * 0.7
    ax.plot([x1, x2], [y1, y2], color=MEMBER_COLOR, linewidth=brace_lw)
    ax.plot([x1, x2], [y2, y1], color=MEMBER_COLOR, linewidth=brace_lw)
    if tracker:
        tracker.record("brace", (x1, y1, x2, y2), label,
                       {**(metadata or {}), "brace_type": "X"})


def draw_brace_v(ax, x1, y1, x2, y2, lw=None, tracker: GroundingTracker = None,
                 label: str = "", metadata: dict = None):
    """Draw V-bracing (chevron) in a rectangular bay."""
    brace_lw = (lw or MEMBER_LW) * 0.7
    mid_x = (x1 + x2) / 2
    ax.plot([x1, mid_x], [y2, y1], color=MEMBER_COLOR, linewidth=brace_lw)
    ax.plot([mid_x, x2], [y1, y2], color=MEMBER_COLOR, linewidth=brace_lw)
    if tracker:
        tracker.record("brace", (x1, y1, x2, y2), label,
                       {**(metadata or {}), "brace_type": "V"})


def draw_brace_single(ax, x1, y1, x2, y2, lw=None, tracker: GroundingTracker = None,
                      label: str = "", metadata: dict = None):
    """Draw single diagonal bracing."""
    brace_lw = (lw or MEMBER_LW) * 0.7
    ax.plot([x1, x2], [y1, y2], color=MEMBER_COLOR, linewidth=brace_lw)
    if tracker:
        tracker.record("brace", (x1, y1, x2, y2), label,
                       {**(metadata or {}), "brace_type": "single"})


# ── Annotation Primitives ──

def label_inline(ax, x1, y1, x2, y2, text: str, offset: float = 0,
                 tracker: GroundingTracker = None):
    """Place section label inline on the member (centered)."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    fs = _rand_fontsize("label")
    if abs(dx) > abs(dy):
        tx, ty = mx, my + offset + 3
        t = ax.text(tx, ty, text, ha="center", va="bottom",
                    fontsize=fs, color=LABEL_COLOR, fontfamily=FONT_FAMILY)
    else:
        tx, ty = mx + offset + 3, my
        t = ax.text(tx, ty, text, ha="left", va="center",
                    fontsize=fs, color=LABEL_COLOR, fontfamily=FONT_FAMILY,
                    rotation=90)
    if tracker:
        tracker.record("label", (x1, y1, x2, y2), text)


def label_above(ax, x1, y1, x2, y2, text: str, offset: float = 5,
                tracker: GroundingTracker = None):
    """Place section label above."""
    mx = (x1 + x2) / 2
    my = max(y1, y2) + offset
    ax.text(mx, my, text, ha="center", va="bottom",
            fontsize=_rand_fontsize("label"), color=LABEL_COLOR,
            fontfamily=FONT_FAMILY)
    if tracker:
        tracker.record("label", (mx - 20, my, mx + 20, my + 10), text)


def label_below(ax, x1, y1, x2, y2, text: str, offset: float = 5,
                tracker: GroundingTracker = None):
    """Place section label below."""
    mx = (x1 + x2) / 2
    my = min(y1, y2) - offset
    ax.text(mx, my, text, ha="center", va="top",
            fontsize=_rand_fontsize("label"), color=LABEL_COLOR,
            fontfamily=FONT_FAMILY)
    if tracker:
        tracker.record("label", (mx - 20, my - 10, mx + 20, my), text)


def label_leader(ax, x1, y1, x2, y2, text: str, offset_x: float = 0,
                 offset_y: float = 20, tracker: GroundingTracker = None):
    """Place section label with a leader arrow."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    tx, ty = mx + offset_x, my + offset_y
    ax.annotate(text, xy=(mx, my), xytext=(tx, ty),
                fontsize=_rand_fontsize("label"), color=LABEL_COLOR,
                fontfamily=FONT_FAMILY,
                arrowprops=dict(arrowstyle="-", color=LABEL_COLOR, lw=0.8),
                ha="center", va="bottom")
    if tracker:
        tracker.record("label", (tx - 20, ty, tx + 20, ty + 10), text)


def label_side(ax, x, y, text: str, side: str = "right",
               tracker: GroundingTracker = None):
    """Place label text to the side of a point."""
    ha = "left" if side == "right" else "right"
    offset = 5 if side == "right" else -5
    ax.text(x + offset, y, text, ha=ha, va="center",
            fontsize=_rand_fontsize("label"), color=LABEL_COLOR,
            fontfamily=FONT_FAMILY)
    if tracker:
        tracker.record("label", (x, y - 5, x + offset * 5, y + 5), text)


LABEL_STYLES = ["inline", "above", "below", "leader"]


def place_label(ax, x1, y1, x2, y2, text: str, style: str = None,
                tracker: GroundingTracker = None):
    """Place a label using a random or specified style."""
    if style is None:
        style = random.choice(LABEL_STYLES)
    if style == "inline":
        label_inline(ax, x1, y1, x2, y2, text, tracker=tracker)
    elif style == "above":
        label_above(ax, x1, y1, x2, y2, text, tracker=tracker)
    elif style == "below":
        label_below(ax, x1, y1, x2, y2, text, tracker=tracker)
    elif style == "leader":
        label_leader(ax, x1, y1, x2, y2, text, tracker=tracker)


# ── Dimension Line Primitives ──

def dim_horizontal(ax, x1, x2, y, label: str, side: str = "below",
                   offset: float = 15, tracker: GroundingTracker = None):
    """Draw a horizontal dimension line with arrows and value."""
    y_dim = y - offset if side == "below" else y + offset

    ax.plot([x1, x1], [y, y_dim], color=DIM_COLOR, linewidth=0.5,
            linestyle="--")
    ax.plot([x2, x2], [y, y_dim], color=DIM_COLOR, linewidth=0.5,
            linestyle="--")
    ax.annotate("", xy=(x2, y_dim), xytext=(x1, y_dim),
                arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=DIM_LW))

    mx = (x1 + x2) / 2
    text_y = y_dim - 4 if side == "below" else y_dim + 2
    va = "top" if side == "below" else "bottom"
    ax.text(mx, text_y, label, ha="center", va=va,
            fontsize=_rand_fontsize("dim"), color=DIM_COLOR,
            fontfamily=FONT_FAMILY)

    if tracker:
        tracker.record("dim_line", (x1, y_dim, x2, y_dim), label)


def dim_vertical(ax, y1, y2, x, label: str, side: str = "left",
                 offset: float = 15, tracker: GroundingTracker = None):
    """Draw a vertical dimension line with arrows and value."""
    x_dim = x - offset if side == "left" else x + offset

    ax.plot([x, x_dim], [y1, y1], color=DIM_COLOR, linewidth=0.5,
            linestyle="--")
    ax.plot([x, x_dim], [y2, y2], color=DIM_COLOR, linewidth=0.5,
            linestyle="--")
    ax.annotate("", xy=(x_dim, y2), xytext=(x_dim, y1),
                arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=DIM_LW))

    my = (y1 + y2) / 2
    ax.text(x_dim - 3, my, label, ha="right", va="center",
            fontsize=_rand_fontsize("dim"), color=DIM_COLOR,
            fontfamily=FONT_FAMILY, rotation=90)

    if tracker:
        tracker.record("dim_line", (x_dim, y1, x_dim, y2), label)


def dim_span_summary(ax, x_start, x_end, y, n_bays: int, spacing: float,
                     side: str = "below", offset: float = 30,
                     tracker: GroundingTracker = None):
    """Draw a summary dimension like '5 x 200 mm spans'."""
    y_dim = y - offset if side == "below" else y + offset

    ax.annotate("", xy=(x_end, y_dim), xytext=(x_start, y_dim),
                arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=DIM_LW))

    mx = (x_start + x_end) / 2
    unit = random.choice(["mm", "m"])
    if unit == "m":
        val = spacing / 1000
        label = f"{n_bays} x {val:.1f} m spans" if val != int(val) else f"{n_bays} x {int(val)} m spans"
    else:
        label = f"{n_bays} x {int(spacing)} mm spans"

    text_y = y_dim - 4 if side == "below" else y_dim + 2
    va = "top" if side == "below" else "bottom"
    ax.text(mx, text_y, label, ha="center", va=va,
            fontsize=_rand_fontsize("dim") + 1, color=DIM_COLOR,
            fontfamily=FONT_FAMILY, fontweight="bold")

    if tracker:
        tracker.record("dim_line", (x_start, y_dim, x_end, y_dim), label,
                       {"n_bays": n_bays, "spacing_mm": spacing})


# ── Figure Setup ──

def create_figure(width_in: float = 12, height_in: float = 8,
                  dpi: int = 150) -> tuple:
    """Create a clean white figure for drawing."""
    fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in), dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save_figure(fig, filepath: str | Path, pad: float = 0.5):
    """Save figure with tight bounding box."""
    fig.savefig(str(filepath), bbox_inches="tight", pad_inches=pad,
                facecolor=BG_COLOR, dpi=fig.dpi)
    plt.close(fig)
