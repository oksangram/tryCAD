"""
Synthetic structural drawing generator with grounding annotations.

Generates training images for Qwen3-VL fine-tuning with:
- Pixel bounding boxes for each structural element (2D grounding)
- Real-world coordinates (mm) for grid lines, members, joints
- AISC section assignments by role

Three levels:
  Level 1: Single member (section label + dimension + bbox)
  Level 2: Simple frames (grid extraction + multi-member grounding)
  Level 3: Composite frames (multi-view + full coordinate output)

Usage:
    python -m extraction_training.synthetic_drawings --level all --output data/extraction
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from extraction_training.sections_pool import (
    pick_section, pick_section_set,
)
from extraction_training.drawing_utils import (
    create_figure, save_figure, GroundingTracker,
    draw_member_line, draw_brace_x, draw_brace_v, draw_brace_single,
    place_label, label_above, label_below, label_side, label_inline,
    dim_horizontal, dim_vertical, dim_span_summary,
    LABEL_COLOR,
)


# ═══════════════════════════════════════════════════════════════════
# LEVEL 1: Single Member Drawings
# ═══════════════════════════════════════════════════════════════════

def generate_level1_horizontal(idx: int) -> tuple[dict, callable]:
    """Single horizontal member with section label and dimension."""
    length_mm = random.choice([1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000])
    section = pick_section(random.choice(["beam", "column", "stringer"]))
    qty = random.choice([1, 1, 1, 2, 2, 3])
    role = random.choice(["beam", "column", "stringer", "member"])
    label_style = random.choice(["inline", "above", "below", "leader"])

    unit = random.choice(["mm", "m"])
    if unit == "m":
        dim_label = f"{length_mm / 1000:.1f} m"
        if length_mm % 1000 == 0:
            dim_label = f"{length_mm // 1000} m"
    else:
        dim_label = f"{length_mm} mm"

    section_text = f"{qty} x {section}" if qty > 1 else section
    if random.random() < 0.3:
        section_text += f" - {dim_label} long"

    ground_truth = {
        "members": [{
            "section": section,
            "quantity": qty,
            "length_mm": length_mm,
            "role": role,
            "start_mm": {"x": 0, "y": 0, "z": 0},
            "end_mm": {"x": length_mm, "y": 0, "z": 0},
        }],
    }

    scale = 0.1

    def render(fig, ax):
        tracker = GroundingTracker()
        x1, x2 = 50, 50 + length_mm * scale
        y = 100

        draw_member_line(ax, x1, y, x2, y, tracker=tracker,
                         elem_type=role, label=section,
                         metadata={"section": section, "length_mm": length_mm})
        place_label(ax, x1, y, x2, y, section_text, style=label_style,
                    tracker=tracker)
        dim_horizontal(ax, x1, x2, y, dim_label, side="below", offset=20,
                       tracker=tracker)

        ax.set_xlim(20, x2 + 50)
        ax.set_ylim(50, 160)

        # Store scale info
        ground_truth["_scale"] = scale
        ground_truth["_tracker"] = tracker

    return ground_truth, render


def generate_level1_vertical(idx: int) -> tuple[dict, callable]:
    """Single vertical member with section label and dimension."""
    height_mm = random.choice([2000, 3000, 4000, 5000, 6000, 8000])
    section = pick_section(random.choice(["column", "column_hss"]))
    qty = random.choice([1, 1, 2])
    label_style = random.choice(["inline", "above", "leader"])

    unit = random.choice(["mm", "m"])
    if unit == "m":
        dim_label = f"{height_mm / 1000:.1f} m"
        if height_mm % 1000 == 0:
            dim_label = f"{height_mm // 1000} m"
    else:
        dim_label = f"{height_mm} mm"

    section_text = f"{qty} x {section}" if qty > 1 else section

    ground_truth = {
        "members": [{
            "section": section,
            "quantity": qty,
            "length_mm": height_mm,
            "role": "column",
            "start_mm": {"x": 0, "y": 0, "z": 0},
            "end_mm": {"x": 0, "y": height_mm, "z": 0},
        }],
    }

    scale = 0.05

    def render(fig, ax):
        tracker = GroundingTracker()
        x = 100
        y1, y2 = 30, 30 + height_mm * scale

        draw_member_line(ax, x, y1, x, y2, tracker=tracker,
                         elem_type="column", label=section,
                         metadata={"section": section, "length_mm": height_mm})
        label_side(ax, x, (y1 + y2) / 2, section_text, side="right",
                   tracker=tracker)
        dim_vertical(ax, y1, y2, x, dim_label, side="left", offset=20,
                     tracker=tracker)

        ax.set_xlim(40, 220)
        ax.set_ylim(10, y2 + 30)

        ground_truth["_scale"] = scale
        ground_truth["_tracker"] = tracker

    return ground_truth, render


def generate_level1(idx: int) -> tuple[dict, callable]:
    if random.random() < 0.6:
        return generate_level1_horizontal(idx)
    else:
        return generate_level1_vertical(idx)


# ═══════════════════════════════════════════════════════════════════
# LEVEL 2: Simple Frame Drawings
# ═══════════════════════════════════════════════════════════════════

def generate_portal_frame(idx: int) -> tuple[dict, callable]:
    """Portal frame elevation: columns + top beam, 1-4 bays."""
    n_bays = random.randint(1, 4)
    bay_spacing_mm = random.choice([3000, 4000, 5000, 6000, 8000])
    height_mm = random.choice([3000, 4000, 5000, 6000, 8000])

    sections = pick_section_set()
    col_section = sections["column"]
    beam_section = sections["beam"]

    # Compute real-world grid
    grid_x = [i * bay_spacing_mm for i in range(n_bays + 1)]

    ground_truth = {
        "drawing_type": "elevation",
        "structure_type": "portal_frame",
        "grid": {
            "x_lines_mm": grid_x,
            "elevations_mm": [0, height_mm],
        },
        "columns": [
            {"grid_x_mm": x, "base_mm": 0, "top_mm": height_mm,
             "section": col_section}
            for x in grid_x
        ],
        "beams": [
            {"start_mm": {"x": grid_x[i], "y": height_mm},
             "end_mm": {"x": grid_x[i + 1], "y": height_mm},
             "section": beam_section, "role": "main_beam"}
            for i in range(n_bays)
        ],
    }

    def render(fig, ax):
        tracker = GroundingTracker()
        scale = 0.06
        margin = 70
        total_w = n_bays * bay_spacing_mm * scale
        h = height_mm * scale

        # Draw columns
        for i, x_mm in enumerate(grid_x):
            x = margin + x_mm * scale
            draw_member_line(ax, x, margin, x, margin + h, tracker=tracker,
                             elem_type="column", label=col_section,
                             metadata={"grid_x_mm": x_mm, "section": col_section})
            if i == 0:
                label_side(ax, x, margin + h / 2, col_section, side="left",
                           tracker=tracker)
            elif i == n_bays:
                label_side(ax, x, margin + h / 2, col_section, side="right",
                           tracker=tracker)

        # Draw top beams
        for i in range(n_bays):
            x1 = margin + grid_x[i] * scale
            x2 = margin + grid_x[i + 1] * scale
            draw_member_line(ax, x1, margin + h, x2, margin + h, tracker=tracker,
                             elem_type="beam", label=beam_section,
                             metadata={"section": beam_section})
        label_above(ax, margin, margin + h, margin + total_w, margin + h,
                    beam_section, offset=8, tracker=tracker)

        # Bay dimensions
        for i in range(n_bays):
            x1 = margin + grid_x[i] * scale
            x2 = margin + grid_x[i + 1] * scale
            val = bay_spacing_mm
            unit = "mm" if val < 10000 else "m"
            v = val if unit == "mm" else val / 1000
            lbl = f"{int(v)} {unit}" if v == int(v) else f"{v:.1f} {unit}"
            dim_horizontal(ax, x1, x2, margin, lbl, side="below",
                           offset=18, tracker=tracker)

        # Height dimension (offset further left to avoid column label)
        unit = "mm" if height_mm < 10000 else "m"
        v = height_mm if unit == "mm" else height_mm / 1000
        h_lbl = f"{int(v)} {unit}" if v == int(v) else f"{v:.1f} {unit}"
        dim_vertical(ax, margin, margin + h, margin, h_lbl, side="left",
                     offset=40, tracker=tracker)

        ax.set_xlim(0, margin + total_w + 80)
        ax.set_ylim(margin - 50, margin + h + 60)

        ground_truth["_scale"] = scale
        ground_truth["_tracker"] = tracker

    return ground_truth, render


def generate_braced_frame(idx: int) -> tuple[dict, callable]:
    """Braced frame elevation: portal frame + bracing in bays."""
    n_bays = random.randint(2, 5)
    bay_spacing_mm = random.choice([3000, 4000, 5000, 6000])
    height_mm = random.choice([3000, 4000, 5000, 6000])
    brace_type = random.choice(["X", "V", "single"])

    sections = pick_section_set()
    col_section = sections["column"]
    beam_section = sections["beam"]
    brace_section = sections["bracing"]

    grid_x = [i * bay_spacing_mm for i in range(n_bays + 1)]

    ground_truth = {
        "drawing_type": "elevation",
        "structure_type": "braced_frame",
        "grid": {
            "x_lines_mm": grid_x,
            "elevations_mm": [0, height_mm],
        },
        "columns": [
            {"grid_x_mm": x, "base_mm": 0, "top_mm": height_mm,
             "section": col_section}
            for x in grid_x
        ],
        "beams": [
            {"start_mm": {"x": grid_x[i], "y": height_mm},
             "end_mm": {"x": grid_x[i + 1], "y": height_mm},
             "section": beam_section, "role": "main_beam"}
            for i in range(n_bays)
        ],
        "bracing": [
            {"bay_x_mm": [grid_x[i], grid_x[i + 1]],
             "elevation_mm": [0, height_mm],
             "type": brace_type, "section": brace_section}
            for i in range(n_bays)
        ],
    }

    def render(fig, ax):
        tracker = GroundingTracker()
        scale = 0.06
        margin = 70
        total_w = n_bays * bay_spacing_mm * scale
        h = height_mm * scale

        # Columns
        for i, x_mm in enumerate(grid_x):
            x = margin + x_mm * scale
            draw_member_line(ax, x, margin, x, margin + h, tracker=tracker,
                             elem_type="column", label=col_section,
                             metadata={"grid_x_mm": x_mm, "section": col_section})

        label_side(ax, margin, margin + h / 2, col_section, side="left",
                   tracker=tracker)

        # Top & bottom beams
        draw_member_line(ax, margin, margin + h, margin + total_w, margin + h,
                         tracker=tracker, elem_type="beam", label=beam_section)
        draw_member_line(ax, margin, margin, margin + total_w, margin,
                         tracker=tracker, elem_type="beam")
        label_above(ax, margin, margin + h, margin + total_w, margin + h,
                    beam_section, offset=8, tracker=tracker)

        # Bracing
        brace_func = {"X": draw_brace_x, "V": draw_brace_v,
                       "single": draw_brace_single}[brace_type]
        for i in range(n_bays):
            x1 = margin + grid_x[i] * scale
            x2 = margin + grid_x[i + 1] * scale
            brace_func(ax, x1, margin, x2, margin + h, tracker=tracker,
                       label=brace_section,
                       metadata={"section": brace_section})

        # Brace label (first bay, white background for readability)
        mid_x = margin + bay_spacing_mm * scale / 2
        mid_y = margin + h / 2
        ax.text(mid_x, mid_y, brace_section, ha="center", va="center",
                fontsize=10, color=LABEL_COLOR, fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))

        # Span summary (offset further below to avoid bay dims)
        dim_span_summary(ax, margin, margin + total_w, margin,
                         n_bays, bay_spacing_mm, side="below", offset=30,
                         tracker=tracker)

        # Height (offset further left to avoid column label)
        unit = "mm" if height_mm < 10000 else "m"
        v = height_mm if unit == "mm" else height_mm / 1000
        h_lbl = f"{int(v)} {unit}" if v == int(v) else f"{v:.1f} {unit}"
        dim_vertical(ax, margin, margin + h, margin, h_lbl, side="left",
                     offset=45, tracker=tracker)

        ax.set_xlim(0, margin + total_w + 80)
        ax.set_ylim(margin - 65, margin + h + 60)

        ground_truth["_scale"] = scale
        ground_truth["_tracker"] = tracker

    return ground_truth, render


def generate_platform_plan(idx: int) -> tuple[dict, callable]:
    """Platform plan view: grid of main + secondary beams."""
    n_main = random.randint(2, 4)
    n_secondary = random.randint(2, 4)
    main_spacing_mm = random.choice([3000, 4000, 5000, 6000])
    sec_spacing_mm = random.choice([1500, 2000, 2500, 3000])

    sections = pick_section_set()
    main_section = sections["beam"]
    sec_section = pick_section("secondary_beam")

    main_lines = [i * main_spacing_mm for i in range(n_main)]
    sec_lines = [j * sec_spacing_mm for j in range(n_secondary)]

    ground_truth = {
        "drawing_type": "plan",
        "structure_type": "platform",
        "grid": {
            "x_lines_mm": sec_lines,
            "z_lines_mm": main_lines,
        },
        "beams": [
            *[{"start_mm": {"x": 0, "z": z}, "end_mm": {"x": sec_lines[-1], "z": z},
               "section": main_section, "role": "main_beam"} for z in main_lines],
            *[{"start_mm": {"x": x, "z": 0}, "end_mm": {"x": x, "z": main_lines[-1]},
               "section": sec_section, "role": "secondary_beam"} for x in sec_lines],
        ],
    }

    def render(fig, ax):
        tracker = GroundingTracker()
        scale = 0.06
        margin = 70
        total_x = (n_secondary - 1) * sec_spacing_mm * scale
        total_z = (n_main - 1) * main_spacing_mm * scale

        # Main beams (horizontal, thicker)
        for i in range(n_main):
            y = margin + i * main_spacing_mm * scale
            draw_member_line(ax, margin, y, margin + total_x, y, lw=2.5,
                             tracker=tracker, elem_type="beam",
                             label=main_section,
                             metadata={"section": main_section, "role": "main_beam"})

        # Secondary beams (vertical, thinner)
        for j in range(n_secondary):
            x = margin + j * sec_spacing_mm * scale
            draw_member_line(ax, x, margin, x, margin + total_z, lw=1.5,
                             tracker=tracker, elem_type="beam",
                             label=sec_section,
                             metadata={"section": sec_section, "role": "secondary_beam"})

        # Labels (with spacing to not overlap dims)
        label_above(ax, margin, margin + total_z,
                    margin + total_x, margin + total_z,
                    f"Main: {main_section}", offset=12, tracker=tracker)

        # Secondary label on left side instead of bottom (avoids dim overlap)
        ax.text(margin - 10, margin + total_z / 2,
                f"Sec: {sec_section}", ha="right", va="center",
                fontsize=10, color=LABEL_COLOR, fontfamily="sans-serif",
                rotation=90)

        # Main beam spacing dims (right side, staggered)
        if n_main <= 4:
            for i in range(n_main - 1):
                y1 = margin + i * main_spacing_mm * scale
                y2 = margin + (i + 1) * main_spacing_mm * scale
                dim_vertical(ax, y1, y2, margin + total_x, f"{main_spacing_mm} mm",
                             side="right", offset=15 + i * 8, tracker=tracker)
        else:
            dim_vertical(ax, margin, margin + total_z, margin + total_x,
                         f"{n_main - 1} x {main_spacing_mm} mm",
                         side="right", offset=15, tracker=tracker)

        # Secondary spacing dims (bottom, use summary)
        dim_span_summary(ax, margin, margin + total_x, margin,
                         n_secondary - 1, sec_spacing_mm, side="below",
                         offset=25, tracker=tracker)

        ax.set_xlim(0, margin + total_x + 100)
        ax.set_ylim(margin - 60, margin + total_z + 60)

        ground_truth["_scale"] = scale
        ground_truth["_tracker"] = tracker

    return ground_truth, render


def generate_piperack_elevation(idx: int) -> tuple[dict, callable]:
    """Piperack elevation: multi-tier portal with stringers."""
    n_bays = random.randint(2, 4)
    bay_spacing_mm = random.choice([4000, 5000, 6000, 8000])
    n_tiers = random.choice([2, 3])
    tier_height_mm = random.choice([3000, 4000, 5000])

    sections = pick_section_set()
    col_section = sections["column"]
    beam_section = sections["beam"]
    stringer_section = sections["stringer"]

    grid_x = [i * bay_spacing_mm for i in range(n_bays + 1)]
    tier_elevations = [(i + 1) * tier_height_mm for i in range(n_tiers)]

    ground_truth = {
        "drawing_type": "elevation",
        "structure_type": "piperack",
        "grid": {
            "x_lines_mm": grid_x,
            "elevations_mm": [0] + tier_elevations,
        },
        "columns": [
            {"grid_x_mm": x, "base_mm": 0, "top_mm": tier_elevations[-1],
             "section": col_section}
            for x in grid_x
        ],
        "beams": [
            {"start_mm": {"x": grid_x[i], "y": elev},
             "end_mm": {"x": grid_x[i + 1], "y": elev},
             "section": beam_section, "role": "tier_beam"}
            for elev in tier_elevations for i in range(n_bays)
        ],
        "stringers": [
            {"elevation_mm": elev, "section": stringer_section}
            for elev in tier_elevations
        ],
    }

    def render(fig, ax):
        tracker = GroundingTracker()
        scale = 0.04
        margin = 70
        total_w = n_bays * bay_spacing_mm * scale
        total_h = n_tiers * tier_height_mm * scale

        # Columns
        for x_mm in grid_x:
            x = margin + x_mm * scale
            draw_member_line(ax, x, margin, x, margin + total_h,
                             tracker=tracker, elem_type="column",
                             label=col_section,
                             metadata={"grid_x_mm": x_mm, "section": col_section})

        label_side(ax, margin, margin + total_h / 2, col_section, side="left",
                   tracker=tracker)

        # Tier beams
        for t, elev in enumerate(tier_elevations):
            y = margin + (t + 1) * tier_height_mm * scale
            draw_member_line(ax, margin, y, margin + total_w, y,
                             tracker=tracker, elem_type="beam",
                             label=beam_section,
                             metadata={"elevation_mm": elev, "section": beam_section})

            # Elevation label (right side)
            unit = "mm" if elev < 10000 else "m"
            v = elev if unit == "mm" else elev / 1000
            elev_str = f"EL +{int(v)} {unit}" if v == int(v) else f"EL +{v:.1f} {unit}"
            ax.text(margin + total_w + 12, y, elev_str,
                    ha="left", va="center", fontsize=9, color=LABEL_COLOR)

        # Beam label on top tier
        label_above(ax, margin, margin + total_h,
                    margin + total_w, margin + total_h,
                    beam_section, offset=8, tracker=tracker)

        # Span summary
        dim_span_summary(ax, margin, margin + total_w, margin,
                         n_bays, bay_spacing_mm, side="below", offset=28,
                         tracker=tracker)

        # Height (offset left to avoid column label)
        total_h_mm = n_tiers * tier_height_mm
        h_lbl = f"{total_h_mm} mm" if total_h_mm < 10000 else f"{total_h_mm // 1000} m"
        dim_vertical(ax, margin, margin + total_h, margin, h_lbl,
                     side="left", offset=45, tracker=tracker)

        ax.set_xlim(0, margin + total_w + 110)
        ax.set_ylim(margin - 60, margin + total_h + 60)

        ground_truth["_scale"] = scale
        ground_truth["_tracker"] = tracker

    return ground_truth, render


LEVEL2_GENERATORS = [
    generate_portal_frame,
    generate_braced_frame,
    generate_platform_plan,
    generate_piperack_elevation,
]


def generate_level2(idx: int) -> tuple[dict, callable]:
    gen = random.choice(LEVEL2_GENERATORS)
    return gen(idx)


# ═══════════════════════════════════════════════════════════════════
# LEVEL 3: Composite Frame Drawings (multi-view)
# ═══════════════════════════════════════════════════════════════════

def generate_composite_frame(idx: int) -> tuple[dict, callable]:
    """Composite: elevation + plan on same sheet, well separated."""
    n_bays = random.randint(2, 4)
    bay_spacing_mm = random.choice([4000, 5000, 6000])
    width_mm = random.choice([3000, 4000, 5000])
    height_mm = random.choice([4000, 5000, 6000])
    brace_type = random.choice(["X", "V", "none"])

    sections = pick_section_set()
    col_section = sections["column"]
    beam_section = sections["beam"]
    stringer_section = sections["stringer"]
    brace_section = sections["bracing"] if brace_type != "none" else None

    grid_x = [i * bay_spacing_mm for i in range(n_bays + 1)]

    members = [
        {"role": "column", "section": col_section},
        {"role": "beam", "section": beam_section},
        {"role": "stringer", "section": stringer_section},
    ]
    if brace_section:
        members.append({"role": "bracing", "section": brace_section,
                         "type": brace_type})

    ground_truth = {
        "views": ["elevation", "plan"],
        "structure_type": "composite_frame",
        "grid": {
            "x_lines_mm": grid_x,
            "z_lines_mm": [0, width_mm],
            "elevations_mm": [0, height_mm],
        },
        "columns": [
            {"grid_x_mm": x, "grid_z_mm": z, "base_mm": 0, "top_mm": height_mm,
             "section": col_section}
            for x in grid_x for z in [0, width_mm]
        ],
        "beams": [
            {"start_mm": {"x": grid_x[i], "y": height_mm},
             "end_mm": {"x": grid_x[i + 1], "y": height_mm},
             "section": beam_section, "role": "main_beam"}
            for i in range(n_bays)
        ],
        "stringers": [
            {"start_mm": {"x": x, "z": 0},
             "end_mm": {"x": x, "z": width_mm},
             "section": stringer_section}
            for x in grid_x
        ],
        "members_summary": members,
    }

    if brace_section:
        ground_truth["bracing"] = [
            {"bay_x_mm": [grid_x[i], grid_x[i + 1]],
             "elevation_mm": [0, height_mm],
             "type": brace_type, "section": brace_section}
            for i in range(n_bays)
        ]

    def render(fig, ax):
        tracker = GroundingTracker()
        scale = 0.04
        margin = 60

        # ── ELEVATION VIEW (top portion) ──
        # Big vertical gap between views
        gap = 80
        plan_height = width_mm * scale
        elev_y_base = margin + plan_height + gap

        total_w = n_bays * bay_spacing_mm * scale
        h = height_mm * scale

        ax.text(margin, elev_y_base + h + 18, "ELEVATION",
                fontsize=13, fontweight="bold", color="black")

        # Columns
        for x_mm in grid_x:
            x = margin + x_mm * scale
            draw_member_line(ax, x, elev_y_base, x, elev_y_base + h,
                             tracker=tracker, elem_type="column",
                             label=col_section,
                             metadata={"grid_x_mm": x_mm})

        label_side(ax, margin, elev_y_base + h / 2, col_section, side="left",
                   tracker=tracker)

        # Top & bottom beams
        draw_member_line(ax, margin, elev_y_base + h,
                         margin + total_w, elev_y_base + h,
                         tracker=tracker, elem_type="beam", label=beam_section)
        draw_member_line(ax, margin, elev_y_base,
                         margin + total_w, elev_y_base,
                         tracker=tracker, elem_type="beam")
        label_above(ax, margin, elev_y_base + h,
                    margin + total_w, elev_y_base + h, beam_section,
                    offset=8, tracker=tracker)

        # Bracing in elevation
        if brace_type != "none":
            brace_func = {"X": draw_brace_x, "V": draw_brace_v}[brace_type]
            for i in range(n_bays):
                x1 = margin + grid_x[i] * scale
                x2 = margin + grid_x[i + 1] * scale
                brace_func(ax, x1, elev_y_base, x2, elev_y_base + h,
                           tracker=tracker, label=brace_section,
                           metadata={"section": brace_section})

            mid_x = margin + bay_spacing_mm * scale / 2
            ax.text(mid_x, elev_y_base + h / 2, brace_section,
                    ha="center", va="center", fontsize=9, color=LABEL_COLOR,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.8))

        # Span summary below elevation
        dim_span_summary(ax, margin, margin + total_w, elev_y_base,
                         n_bays, bay_spacing_mm, side="below", offset=25,
                         tracker=tracker)

        # ── PLAN VIEW (bottom portion) ──
        plan_y_base = margin
        w = width_mm * scale

        ax.text(margin, plan_y_base + w + 15, "PLAN",
                fontsize=13, fontweight="bold", color="black")

        # Stringers (along length — horizontal)
        draw_member_line(ax, margin, plan_y_base,
                         margin + total_w, plan_y_base, lw=2.5,
                         tracker=tracker, elem_type="stringer",
                         label=stringer_section)
        draw_member_line(ax, margin, plan_y_base + w,
                         margin + total_w, plan_y_base + w, lw=2.5,
                         tracker=tracker, elem_type="stringer",
                         label=stringer_section)

        # Cross beams at each grid line
        for x_mm in grid_x:
            x = margin + x_mm * scale
            draw_member_line(ax, x, plan_y_base, x, plan_y_base + w, lw=1.5,
                             tracker=tracker, elem_type="beam")

        # Stringer label
        ax.text(margin + total_w + 10, plan_y_base + w / 2, stringer_section,
                ha="left", va="center", fontsize=10, color=LABEL_COLOR)

        # Width dimension (left side, no overlap with elevation)
        dim_vertical(ax, plan_y_base, plan_y_base + w, margin,
                     f"{width_mm} mm", side="left", offset=30,
                     tracker=tracker)

        ax.set_xlim(0, margin + total_w + 110)
        ax.set_ylim(margin - 55, elev_y_base + h + 55)

        ground_truth["_scale"] = scale
        ground_truth["_tracker"] = tracker

    return ground_truth, render


def generate_level3(idx: int) -> tuple[dict, callable]:
    return generate_composite_frame(idx)


# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════

GENERATORS = {1: generate_level1, 2: generate_level2, 3: generate_level3}
DEFAULT_COUNTS = {1: 1000, 2: 1500, 3: 1000}


def generate_dataset(level: int, count: int, output_dir: Path):
    """Generate a dataset of synthetic drawings with grounding data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    gen_func = GENERATORS[level]
    manifest = []

    for i in range(count):
        ground_truth, render_func = gen_func(i)

        fig, ax = create_figure()
        render_func(fig, ax)

        # Save image
        img_name = f"level{level}_{i:05d}.png"
        img_path = images_dir / img_name
        save_figure(fig, img_path)

        # Clean up internal keys from ground_truth
        gt_clean = {k: v for k, v in ground_truth.items()
                    if not k.startswith("_")}

        entry = {
            "image": f"images/{img_name}",
            "ground_truth": gt_clean,
            "level": level,
        }
        manifest.append(entry)

        if (i + 1) % 100 == 0:
            print(f"  Level {level}: {i + 1}/{count} generated")

    manifest_path = output_dir / f"level{level}_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"Level {level}: {count} images saved to {output_dir}")
    print(f"  Manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic structural drawings")
    parser.add_argument("--level", type=str, default="all",
                        help="Level: 1, 2, 3, or 'all'")
    parser.add_argument("--count", type=int, default=None,
                        help="Override default count per level")
    parser.add_argument("--output", type=str, default="data/extraction",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.level == "all":
        levels = [1, 2, 3]
    else:
        levels = [int(args.level)]

    total = 0
    for level in levels:
        count = args.count or DEFAULT_COUNTS[level]
        level_dir = output_dir / f"level{level}"
        print(f"\n{'=' * 60}")
        print(f"Generating Level {level}: {count} images")
        print(f"{'=' * 60}")
        generate_dataset(level, count, level_dir)
        total += count

    print(f"\n✅ Total: {total} images generated in {output_dir}")


if __name__ == "__main__":
    main()
