"""
AISC section pool for synthetic drawing generation.

Provides realistic section names grouped by type and typical usage role.
"""

from __future__ import annotations
import random

# ── AISC Section Database ──

W_SHAPES = [
    "W8X31", "W8X35", "W8X48",
    "W10X22", "W10X33", "W10X49",
    "W12X26", "W12X40", "W12X65", "W12X87",
    "W14X48", "W14X68", "W14X90", "W14X120",
    "W16X36", "W16X57", "W16X77",
    "W18X35", "W18X50", "W18X76",
    "W21X44", "W21X62",
    "W24X55", "W24X76",
]

HSS_SHAPES = [
    "HSS4X4X1/4", "HSS4X4X3/8",
    "HSS6X6X1/4", "HSS6X6X3/8", "HSS6X6X1/2",
    "HSS8X8X3/8", "HSS8X8X1/2",
    "HSS10X10X3/8", "HSS10X10X1/2",
]

L_SHAPES = [
    "L3X3X1/4", "L3X3X3/8",
    "L4X4X1/4", "L4X4X3/8", "L4X4X1/2",
    "L5X5X3/8", "L5X5X1/2",
    "L6X6X3/8", "L6X6X1/2",
]

C_SHAPES = [
    "C8X11.5", "C8X18.75",
    "C10X15.3", "C10X20", "C10X30",
    "C12X20.7", "C12X25",
    "C15X33.9", "C15X40",
]

# ── Role-based section selection ──

ROLE_SECTIONS = {
    "column": {
        "light": ["W8X31", "W8X35", "W10X33", "W10X49"],
        "medium": ["W12X40", "W12X65", "W14X48", "W14X68"],
        "heavy": ["W14X90", "W14X120", "W12X87"],
    },
    "beam": {
        "light": ["W10X22", "W12X26", "W14X48", "W16X36"],
        "medium": ["W16X57", "W18X35", "W18X50", "W21X44"],
        "heavy": ["W21X62", "W24X55", "W24X76", "W18X76"],
    },
    "secondary_beam": {
        "light": ["W8X31", "W10X22", "W12X26"],
        "medium": ["W14X48", "W16X36"],
    },
    "stringer": {
        "light": ["C8X11.5", "C10X15.3"],
        "medium": ["C10X20", "C12X20.7", "C12X25"],
        "heavy": ["C15X33.9", "C15X40"],
    },
    "bracing": {
        "light": ["L3X3X1/4", "L3X3X3/8", "L4X4X1/4"],
        "medium": ["L4X4X3/8", "L5X5X3/8"],
        "heavy": ["L5X5X1/2", "L6X6X3/8", "L6X6X1/2"],
    },
    "column_hss": {
        "light": ["HSS4X4X1/4", "HSS4X4X3/8"],
        "medium": ["HSS6X6X1/4", "HSS6X6X3/8", "HSS6X6X1/2"],
        "heavy": ["HSS8X8X3/8", "HSS8X8X1/2", "HSS10X10X3/8"],
    },
}


def pick_section(role: str, weight: str = None) -> str:
    """Pick a random section appropriate for the given structural role."""
    if role not in ROLE_SECTIONS:
        return random.choice(W_SHAPES)
    pool = ROLE_SECTIONS[role]
    if weight is None:
        weight = random.choice(list(pool.keys()))
    return random.choice(pool.get(weight, pool["medium"]))


def pick_section_set(structure_weight: str = None) -> dict[str, str]:
    """Pick a coherent set of sections for a complete structure."""
    if structure_weight is None:
        structure_weight = random.choice(["light", "medium", "heavy"])
    return {
        "column": pick_section("column", structure_weight),
        "beam": pick_section("beam", structure_weight),
        "bracing": pick_section("bracing", structure_weight),
        "stringer": pick_section("stringer", structure_weight),
    }


ALL_SECTIONS = W_SHAPES + HSS_SHAPES + L_SHAPES + C_SHAPES
