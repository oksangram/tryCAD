"""
Natural language specification generator.

Converts structure parameters into natural language descriptions
at three detail levels: full, summary, and underspecified.
"""

from __future__ import annotations
import random
from .sampler import StructureParams


def generate_nl_spec(params: StructureParams, detail: str = "full") -> str:
    """
    Generate a natural language specification from parameters.

    Args:
        params: Structure parameters.
        detail: "full" (all details), "summary" (key info), "underspecified" (vague).

    Returns:
        Natural language specification string.
    """
    if detail == "full":
        return _full_spec(params)
    elif detail == "summary":
        return _summary_spec(params)
    elif detail == "underspecified":
        return _underspecified_spec(params)
    else:
        return _full_spec(params)


def _full_spec(params: StructureParams) -> str:
    """Detailed specification with all parameters."""
    if params.structure_type == "portal_frame":
        return _full_portal_frame(params)
    elif params.structure_type == "platform":
        return _full_platform(params)
    elif params.structure_type == "pipe_rack":
        return _full_pipe_rack(params)
    return _full_generic(params)


def _full_portal_frame(p: StructureParams) -> str:
    total_length = p.n_bays_x * p.span_x
    total_height = p.n_levels * p.height_per_level
    total_load = p.load_dl_kpa + p.load_ll_kpa

    lines = [
        f"Design a {p.n_bays_x}-bay portal frame with the following specifications:",
        f"",
        f"Geometry:",
        f"- Number of bays: {p.n_bays_x}",
        f"- Bay span: {p.span_x}m (total length: {total_length}m)",
        f"- Number of levels: {p.n_levels}",
        f"- Height per level: {p.height_per_level}m (total height: {total_height}m)",
        f"",
        f"Sections:",
        f"- Columns: {p.column_section}",
        f"- Beams: {p.beam_section_x}",
    ]

    if p.bracing_type != "NONE":
        bay_desc = ", ".join(f"bay {b+1}" for b in p.bracing_bays)
        lines.extend([
            f"- Bracing: {p.bracing_type}-type in {bay_desc}",
            f"- Bracing section: {p.bracing_section}",
        ])

    lines.extend([
        f"",
        f"Supports: {p.support_type} at all base joints",
        f"",
        f"Loading:",
        f"- Dead load: {p.load_dl_kpa} kPa",
        f"- Live load: {p.load_ll_kpa} kPa",
        f"- Tributary width: {p.tributary_width}m",
        f"- Total UDL on beams: {total_load * p.tributary_width} kN/m",
    ])

    return "\n".join(lines)


def _full_platform(p: StructureParams) -> str:
    total_x = p.n_bays_x * p.span_x
    total_z = p.n_bays_z * p.span_z
    total_height = p.n_levels * p.height_per_level
    total_load = p.load_dl_kpa + p.load_ll_kpa

    lines = [
        f"Design a {p.n_bays_x}x{p.n_bays_z} structural platform:",
        f"",
        f"Geometry:",
        f"- Bays in X: {p.n_bays_x} at {p.span_x}m spacing (total: {total_x}m)",
        f"- Bays in Z: {p.n_bays_z} at {p.span_z}m spacing (total: {total_z}m)",
        f"- Levels: {p.n_levels} at {p.height_per_level}m height (total: {total_height}m)",
        f"",
        f"Sections:",
        f"- Columns: {p.column_section}",
        f"- Main beams (X): {p.beam_section_x}",
        f"- Main beams (Z): {p.beam_section_z}",
    ]

    if p.bracing_type != "NONE":
        lines.append(f"- Bracing: {p.bracing_type}-type, section {p.bracing_section}")

    lines.extend([
        f"",
        f"Supports: {p.support_type} at all base joints",
        f"",
        f"Loading:",
        f"- Dead + Live: {total_load} kPa on top level",
    ])

    if p.equipment_loads:
        lines.append(f"- Equipment loads:")
        for i, eq in enumerate(p.equipment_loads, 1):
            lines.append(f"  {i}. {eq['weight_kn']} kN at X={eq['x']}m, Z={eq['z']}m")

    return "\n".join(lines)


def _full_pipe_rack(p: StructureParams) -> str:
    total_x = p.n_bays_x * p.span_x
    total_z = p.n_bays_z * p.span_z
    total_height = p.n_levels * p.height_per_level

    lines = [
        f"Design a {p.n_bays_x}-bay pipe rack ({p.n_levels} tier{'s' if p.n_levels > 1 else ''}):",
        f"",
        f"Geometry:",
        f"- Number of bays: {p.n_bays_x} at {p.span_x}m spacing (total: {total_x}m)",
        f"- Rack width: {p.span_z}m",
        f"- Tiers: {p.n_levels} at {p.height_per_level}m per tier (total: {total_height}m)",
        f"",
        f"Sections:",
        f"- Columns: {p.column_section}",
        f"- Longitudinal beams: {p.beam_section_x}",
        f"- Transverse beams: {p.beam_section_z}",
        f"- Bracing: {p.bracing_type}-type ({p.bracing_section}) in end bays",
        f"",
        f"Supports: {p.support_type}",
        f"",
        f"Loading:",
        f"- Pipe/cable tray dead load: {p.load_dl_kpa} kPa",
        f"- Operating/maintenance live load: {p.load_ll_kpa} kPa",
    ]

    return "\n".join(lines)


def _full_generic(p: StructureParams) -> str:
    return f"Design a {p.structure_type} structure with {p.n_bays_x} bays at {p.span_x}m span, {p.height_per_level}m height, {p.column_section} columns."


def _summary_spec(p: StructureParams) -> str:
    """Concise specification with key parameters only."""
    templates = {
        "portal_frame": [
            f"Design a {p.n_bays_x}-bay portal frame: {p.span_x}m spans, {p.height_per_level}m height, {p.column_section} columns, {p.beam_section_x} beams, {p.support_type} base. Load: {p.load_dl_kpa + p.load_ll_kpa} kPa.",
            f"{p.n_bays_x}-bay frame, {p.span_x}m bay spacing, {p.height_per_level * p.n_levels}m total height. Columns: {p.column_section}, beams: {p.beam_section_x}. {p.bracing_type} bracing in end bays. DL+LL = {p.load_dl_kpa + p.load_ll_kpa} kPa.",
        ],
        "platform": [
            f"Design a {p.n_bays_x}x{p.n_bays_z} platform, {p.span_x}m x {p.span_z}m bays, {p.height_per_level}m height. Columns {p.column_section}, beams {p.beam_section_x}. Load: {p.load_dl_kpa + p.load_ll_kpa} kPa.",
            f"Structural platform: {p.n_bays_x * p.span_x}m x {p.n_bays_z * p.span_z}m footprint, {p.height_per_level}m high. {p.column_section} columns, {p.beam_section_x} beams. {p.support_type} supports.",
        ],
        "pipe_rack": [
            f"{p.n_bays_x}-bay pipe rack, {p.n_levels} tiers, {p.span_x}m spacing, {p.span_z}m width, {p.height_per_level}m per tier. {p.column_section} columns, {p.bracing_type} bracing.",
        ],
    }

    choices = templates.get(p.structure_type, [_full_generic(p)])
    return random.choice(choices)


def _underspecified_spec(p: StructureParams) -> str:
    """Deliberately vague specification that requires clarification."""
    templates = {
        "portal_frame": [
            f"I need a portal frame about {p.n_bays_x * p.span_x}m long and {p.height_per_level}m high.",
            f"Design me a frame structure. It should be around {p.n_bays_x} bays wide.",
            f"I need a steel frame for a {_random_use_case_frame()}. Roughly {p.n_bays_x * p.span_x}m x {p.height_per_level}m.",
        ],
        "platform": [
            f"I need a platform about {p.n_bays_x * p.span_x}m by {p.n_bays_z * p.span_z}m, around {p.height_per_level}m high.",
            f"Design a steel platform for {_random_use_case_platform()}.",
            f"We need an access platform, approximately {p.n_bays_x * p.span_x}m long.",
        ],
        "pipe_rack": [
            f"I need a pipe rack for {_random_pipe_count()} pipes, about {p.n_bays_x * p.span_x}m long.",
            f"Design a pipe rack structure. We have {p.n_levels} tier{'s' if p.n_levels > 1 else ''} of piping to support.",
        ],
    }

    choices = templates.get(p.structure_type, [f"Design a {p.structure_type} structure."])
    return random.choice(choices)


def _random_use_case_frame() -> str:
    return random.choice([
        "warehouse extension", "workshop", "loading bay",
        "equipment shelter", "storage area", "maintenance bay",
    ])


def _random_use_case_platform() -> str:
    return random.choice([
        "vessel maintenance access", "compressor deck",
        "equipment module", "pump access", "valve station",
        "electrical switchgear area",
    ])


def _random_pipe_count() -> str:
    return random.choice(["8-12", "15-20", "6-10", "20-30", "10-15"])
