"""
Design reasoning trace generator.

Creates step-by-step <think> blocks showing the engineering design process
for a given structure. These traces teach the model HOW to design, not
just what to output.
"""

from __future__ import annotations
from .sampler import StructureParams


def generate_reasoning(params: StructureParams) -> str:
    """
    Generate a design reasoning trace for the given parameters.

    Returns:
        A <think>...</think> block with step-by-step reasoning.
    """
    if params.structure_type == "portal_frame":
        return _reasoning_portal_frame(params)
    elif params.structure_type == "platform":
        return _reasoning_platform(params)
    elif params.structure_type == "pipe_rack":
        return _reasoning_pipe_rack(params)
    return _reasoning_generic(params)


def _reasoning_portal_frame(p: StructureParams) -> str:
    total_x = p.n_bays_x * p.span_x
    total_h = p.n_levels * p.height_per_level
    total_load = p.load_dl_kpa + p.load_ll_kpa
    udl = total_load * p.tributary_width

    grid_x = [i * p.span_x for i in range(p.n_bays_x + 1)]

    lines = [
        "<think>",
        f"STRUCTURE TYPE: {p.n_bays_x}-bay portal frame, {p.n_levels} level(s)",
        "",
        "STEP 1 — GEOMETRY LAYOUT:",
        f"  Total length: {p.n_bays_x} bays × {p.span_x}m = {total_x}m",
        f"  Total height: {p.n_levels} levels × {p.height_per_level}m = {total_h}m",
        f"  Column grid in X: {grid_x}",
        f"  Frame is planar (2D in XY plane, Z = 0)",
        f"  → Need {p.n_bays_x + 1} column lines",
        "",
        "STEP 2 — COLUMNS:",
        f"  {p.n_bays_x + 1} columns from Y=0 to Y={total_h}",
    ]

    if p.n_levels > 1:
        for lvl in range(p.n_levels):
            lines.append(f"  Level {lvl+1}: Y={lvl * p.height_per_level} to Y={(lvl+1) * p.height_per_level}")

    lines.extend([
        f"  Section: {p.column_section}",
        f"  → create_column_grid(grid_x={grid_x}, grid_z=[0], base_y=0, top_y={total_h}, section='{p.column_section}')",
        "",
        "STEP 3 — BEAMS:",
        f"  Horizontal beams at each level top connecting column tops",
        f"  {p.n_bays_x} beam spans per level × {p.n_levels} levels = {p.n_bays_x * p.n_levels} beams",
        f"  Section: {p.beam_section_x} (suitable for {p.span_x}m span)",
        f"  → create_beam_spans at each level elevation",
    ])

    if p.bracing_type != "NONE":
        bay_desc = ", ".join(str(b + 1) for b in p.bracing_bays)
        lines.extend([
            "",
            "STEP 4 — BRACING:",
            f"  Type: {p.bracing_type}-bracing",
            f"  Bays: {bay_desc}",
            f"  Diagonal length ≈ √({p.span_x}² + {p.height_per_level}²) = {(p.span_x**2 + p.height_per_level**2)**0.5:.1f}m",
            f"  Section: {p.bracing_section}",
            f"  → add_bracing in bay(s) {bay_desc}",
        ])

    lines.extend([
        "",
        f"STEP 5 — SUPPORTS:",
        f"  {p.support_type} at all {p.n_bays_x + 1} base joints (Y=0)",
        f"  → add_supports(type='{p.support_type}')",
        "",
        f"STEP 6 — LOADS:",
        f"  DL = {p.load_dl_kpa} kPa, LL = {p.load_ll_kpa} kPa",
        f"  Tributary width = {p.tributary_width}m",
        f"  UDL = ({p.load_dl_kpa} + {p.load_ll_kpa}) × {p.tributary_width} = {udl} kN/m (downward → negative GY)",
        f"  Applied to all beam members at top level",
        f"  → add_member_load(UNI, GY, {-udl})",
        "",
        f"STEP 7 — ASSEMBLE & VALIDATE",
        "</think>",
    ])

    return "\n".join(lines)


def _reasoning_platform(p: StructureParams) -> str:
    total_x = p.n_bays_x * p.span_x
    total_z = p.n_bays_z * p.span_z
    total_h = p.n_levels * p.height_per_level
    total_load = p.load_dl_kpa + p.load_ll_kpa

    grid_x = [i * p.span_x for i in range(p.n_bays_x + 1)]
    grid_z = [i * p.span_z for i in range(p.n_bays_z + 1)]
    n_cols = (p.n_bays_x + 1) * (p.n_bays_z + 1)

    lines = [
        "<think>",
        f"STRUCTURE TYPE: {p.n_bays_x}×{p.n_bays_z} structural platform, {p.n_levels} level(s)",
        "",
        "STEP 1 — FOOTPRINT:",
        f"  X direction: {p.n_bays_x} bays × {p.span_x}m = {total_x}m",
        f"  Z direction: {p.n_bays_z} bays × {p.span_z}m = {total_z}m",
        f"  Height: {p.n_levels} × {p.height_per_level}m = {total_h}m",
        f"  Column grid X: {grid_x}",
        f"  Column grid Z: {grid_z}",
        f"  → {p.n_bays_x + 1} × {p.n_bays_z + 1} = {n_cols} columns per level",
        "",
        "STEP 2 — COLUMNS:",
        f"  {n_cols} columns, section: {p.column_section}",
        f"  Tributary area per column: {p.span_x}m × {p.span_z}m = {p.span_x * p.span_z}m²",
        f"  → create_column_grid(grid_x={grid_x}, grid_z={grid_z})",
        "",
        "STEP 3 — BEAM GRID:",
        f"  Beam grid at elevation {total_h}m",
        f"  Main beams in X: {p.beam_section_x}",
        f"  Main beams in Z: {p.beam_section_z}",
        f"  → create_beam_grid at each level elevation",
    ]

    if p.bracing_type != "NONE":
        lines.extend([
            "",
            "STEP 4 — BRACING:",
            f"  {p.bracing_type}-bracing at front (Z=0) and back (Z={total_z}) faces",
            f"  Section: {p.bracing_section}",
        ])

    lines.extend([
        "",
        f"STEP 5 — SUPPORTS:",
        f"  {p.support_type} at all {n_cols} base joints",
        "",
        f"STEP 6 — LOADS:",
        f"  Area load: {total_load} kPa on top level beams",
    ])

    if p.equipment_loads:
        lines.append(f"  Equipment loads:")
        for eq in p.equipment_loads:
            lines.append(f"    - {eq['weight_kn']} kN at ({eq['x']}, {eq['z']})")

    lines.extend([
        "",
        f"STEP 7 — ASSEMBLE & VALIDATE",
        "</think>",
    ])

    return "\n".join(lines)


def _reasoning_pipe_rack(p: StructureParams) -> str:
    total_x = p.n_bays_x * p.span_x

    lines = [
        "<think>",
        f"STRUCTURE TYPE: {p.n_bays_x}-bay pipe rack, {p.n_levels} tier(s)",
        "",
        "STEP 1 — LAYOUT:",
        f"  Total length: {p.n_bays_x} × {p.span_x}m = {total_x}m",
        f"  Width: {p.span_z}m",
        f"  Tiers: {p.n_levels} at {p.height_per_level}m each",
        "",
        "STEP 2 — COLUMNS:",
        f"  {(p.n_bays_x + 1) * 2} columns (both sides)",
        f"  Section: {p.column_section}",
        "",
        "STEP 3 — BEAM GRID AT EACH TIER:",
        f"  Longitudinal beams: {p.beam_section_x}",
        f"  Transverse beams: {p.beam_section_z}",
        "",
        "STEP 4 — VERTICAL BRACING:",
        f"  {p.bracing_type}-bracing in end bays (both faces)",
        f"  Section: {p.bracing_section}",
        "",
        f"STEP 5 — SUPPORTS: {p.support_type}",
        "",
        f"STEP 6 — PIPE LOADS: {p.load_dl_kpa + p.load_ll_kpa} kPa on each tier",
        "",
        "STEP 7 — ASSEMBLE & VALIDATE",
        "</think>",
    ]

    return "\n".join(lines)


def _reasoning_generic(p: StructureParams) -> str:
    return (
        "<think>\n"
        f"Structure: {p.structure_type}\n"
        f"Grid: {p.n_bays_x}x{p.n_bays_z}, spans: {p.span_x}x{p.span_z}m\n"
        f"Height: {p.height_per_level}m\n"
        "Plan: columns → beams → bracing → supports → loads → assemble\n"
        "</think>"
    )
