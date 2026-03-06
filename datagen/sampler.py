"""
Engineering-aware parameter sampler.

Generates realistic parameter combinations for structural design,
ensuring values fall within engineering-appropriate ranges.
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Optional

from standards.design_rules import (
    select_column_section, select_beam_section, select_bracing_section,
)


@dataclass
class StructureParams:
    """Complete parameter set for a structural design."""
    structure_type: str              # "portal_frame", "platform", "pipe_rack", etc.
    n_bays_x: int                    # Number of bays in X
    n_bays_z: int                    # Number of bays in Z (1 for planar frames)
    n_levels: int                    # Number of levels
    span_x: float                    # Bay span in X (meters)
    span_z: float                    # Bay span in Z (meters)
    height_per_level: float          # Height per level (meters)
    base_elevation: float            # Base elevation (meters)
    column_section: str              # AISC section for columns
    beam_section_x: str              # Main beams in X direction
    beam_section_z: str              # Main beams in Z direction
    bracing_type: str                # "X", "K", "V", "CHEVRON", or "NONE"
    bracing_bays: list[int]          # Which X-bays have bracing (0-indexed)
    bracing_section: str             # Bracing section
    support_type: str                # "FIXED" or "PINNED"
    load_dl_kpa: float               # Dead load (kPa)
    load_ll_kpa: float               # Live load (kPa)
    tributary_width: float           # Tributary width for UDL conversion (m)
    equipment_loads: list[dict] = field(default_factory=list)  # [{weight_kn, x, z}]
    title: str = ""


def sample_portal_frame() -> StructureParams:
    """Sample parameters for a portal frame."""
    n_bays = random.randint(1, 6)
    span = random.choice([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    height = random.choice([3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0])
    n_levels = random.choices([1, 2, 3], weights=[60, 30, 10])[0]

    trib = random.choice([1.0, 1.5, 2.0])
    col_trib_area = span * trib

    bracing_type = random.choice(["X", "X", "K", "V", "CHEVRON", "NONE"])
    # Bracing in end bays or every other bay
    if bracing_type == "NONE":
        bracing_bays = []
    elif n_bays <= 2:
        bracing_bays = list(range(n_bays))
    else:
        bracing_bays = [0, n_bays - 1]  # end bays

    diag = math.sqrt(span**2 + height**2)

    return StructureParams(
        structure_type="portal_frame",
        n_bays_x=n_bays,
        n_bays_z=0,
        n_levels=n_levels,
        span_x=span,
        span_z=0,
        height_per_level=height,
        base_elevation=0.0,
        column_section=select_column_section(height * n_levels, col_trib_area),
        beam_section_x=select_beam_section(span),
        beam_section_z="",
        bracing_type=bracing_type,
        bracing_bays=bracing_bays,
        bracing_section=select_bracing_section(diag) if bracing_type != "NONE" else "",
        support_type=random.choice(["FIXED", "FIXED", "PINNED"]),
        load_dl_kpa=random.choice([2.0, 3.0, 4.0, 5.0]),
        load_ll_kpa=random.choice([2.0, 3.0, 4.0, 5.0, 7.5, 10.0]),
        tributary_width=trib,
        title=f"{n_bays}-Bay Portal Frame",
    )


def sample_platform() -> StructureParams:
    """Sample parameters for a structural platform."""
    n_bays_x = random.randint(1, 4)
    n_bays_z = random.randint(1, 3)
    span_x = random.choice([3.0, 4.0, 5.0, 6.0, 8.0])
    span_z = random.choice([3.0, 4.0, 5.0, 6.0, 8.0])
    height = random.choice([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    n_levels = random.choices([1, 2], weights=[70, 30])[0]

    col_trib = span_x * span_z
    bracing_type = random.choice(["X", "X", "K", "NONE"])
    diag = math.sqrt(span_x**2 + height**2)

    bracing_bays = []
    if bracing_type != "NONE" and n_bays_x > 0:
        bracing_bays = [0, n_bays_x - 1] if n_bays_x > 1 else [0]

    # Equipment loads
    equip_loads = []
    n_equip = random.choices([0, 1, 2], weights=[40, 40, 20])[0]
    total_x = n_bays_x * span_x
    total_z = n_bays_z * span_z
    for _ in range(n_equip):
        equip_loads.append({
            "weight_kn": random.choice([20, 30, 50, 80, 100, 150, 200]),
            "x": round(random.uniform(span_x * 0.3, total_x - span_x * 0.3), 1),
            "z": round(random.uniform(span_z * 0.3, total_z - span_z * 0.3), 1),
        })

    return StructureParams(
        structure_type="platform",
        n_bays_x=n_bays_x,
        n_bays_z=n_bays_z,
        n_levels=n_levels,
        span_x=span_x,
        span_z=span_z,
        height_per_level=height,
        base_elevation=0.0,
        column_section=select_column_section(height * n_levels, col_trib),
        beam_section_x=select_beam_section(span_x),
        beam_section_z=select_beam_section(span_z),
        bracing_type=bracing_type,
        bracing_bays=bracing_bays,
        bracing_section=select_bracing_section(diag) if bracing_type != "NONE" else "",
        support_type=random.choice(["FIXED", "PINNED"]),
        load_dl_kpa=random.choice([2.0, 3.0, 5.0]),
        load_ll_kpa=random.choice([3.0, 5.0, 7.5, 10.0]),
        tributary_width=span_z,
        equipment_loads=equip_loads,
        title=f"{n_bays_x}x{n_bays_z} Platform ({n_levels} level{'s' if n_levels > 1 else ''})",
    )


def sample_pipe_rack() -> StructureParams:
    """Sample parameters for a pipe rack."""
    n_bays = random.randint(3, 10)
    span = random.choice([6.0, 7.0, 8.0, 9.0, 10.0, 12.0])
    width = random.choice([3.0, 4.0, 5.0, 6.0])
    height = random.choice([4.0, 5.0, 6.0, 7.0, 8.0])
    n_levels = random.choices([1, 2, 3], weights=[30, 50, 20])[0]

    col_trib = span * width
    diag = math.sqrt(span**2 + height**2)

    return StructureParams(
        structure_type="pipe_rack",
        n_bays_x=n_bays,
        n_bays_z=1,
        n_levels=n_levels,
        span_x=span,
        span_z=width,
        height_per_level=height,
        base_elevation=0.0,
        column_section=select_column_section(height * n_levels, col_trib),
        beam_section_x=select_beam_section(span),
        beam_section_z=select_beam_section(width),
        bracing_type=random.choice(["X", "K", "V"]),
        bracing_bays=[0, n_bays - 1],
        bracing_section=select_bracing_section(diag),
        support_type=random.choice(["FIXED", "PINNED"]),
        load_dl_kpa=random.choice([1.0, 2.0, 3.0]),
        load_ll_kpa=random.choice([5.0, 7.5, 10.0, 12.0, 15.0]),
        tributary_width=width,
        title=f"{n_bays}-Bay Pipe Rack ({n_levels} tier{'s' if n_levels > 1 else ''})",
    )


# ── Sampling dispatcher ──

STRUCTURE_SAMPLERS = {
    "portal_frame": sample_portal_frame,
    "platform": sample_platform,
    "pipe_rack": sample_pipe_rack,
}

STRUCTURE_WEIGHTS = {
    "portal_frame": 40,
    "platform": 40,
    "pipe_rack": 20,
}


def sample_random_structure() -> StructureParams:
    """Sample a random structure type with weighted probabilities."""
    types = list(STRUCTURE_WEIGHTS.keys())
    weights = [STRUCTURE_WEIGHTS[t] for t in types]
    chosen = random.choices(types, weights=weights)[0]
    return STRUCTURE_SAMPLERS[chosen]()
