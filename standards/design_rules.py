"""
Engineering design rules for structural section selection and sizing.

These heuristics are used by the data generator and tools to make
engineering-appropriate choices for member sections based on span and load.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


SECTIONS_DIR = Path(__file__).parent / "sections"


# ──────────────────────────────────────────────────────────────
# Section lookup
# ──────────────────────────────────────────────────────────────

_section_cache: Optional[dict] = None


def _load_sections() -> dict:
    global _section_cache
    if _section_cache is None:
        path = SECTIONS_DIR / "aisc_w_shapes.json"
        with open(path) as f:
            _section_cache = json.load(f)
    return _section_cache


def get_section(name: str) -> Optional[dict]:
    """Get section properties by designation (e.g., 'W12X65')."""
    sections = _load_sections()
    return sections.get(name)


def section_exists(name: str) -> bool:
    """Check if a section designation exists in the database."""
    sections = _load_sections()
    return name in sections


def all_section_names() -> list[str]:
    """Return all available section designations."""
    return list(_load_sections().keys())


# ──────────────────────────────────────────────────────────────
# Section selection heuristics
# ──────────────────────────────────────────────────────────────

# Column section lookup: (max_height_m, max_tributary_area_m2) → section
COLUMN_SELECTION = [
    # (max_height, max_trib_area, section)
    (4.0,  20,  "W8X31"),
    (4.0,  40,  "W10X49"),
    (4.0,  80,  "W12X65"),
    (6.0,  20,  "W10X49"),
    (6.0,  40,  "W12X65"),
    (6.0,  80,  "W12X79"),
    (8.0,  30,  "W12X65"),
    (8.0,  60,  "W12X79"),
    (8.0, 120,  "W14X90"),
    (10.0, 40,  "W12X79"),
    (10.0, 80,  "W14X90"),
    (10.0, 200, "W14X120"),
    (15.0, 60,  "W14X90"),
    (15.0, 120, "W14X120"),
    (15.0, 300, "W14X159"),
]

# Beam section lookup: (max_span_m) → section
BEAM_SELECTION = [
    # (max_span, section)
    (3.0,  "W8X18"),
    (4.0,  "W10X22"),
    (5.0,  "W12X26"),
    (6.0,  "W14X30"),
    (7.0,  "W16X36"),
    (8.0,  "W16X40"),
    (9.0,  "W18X46"),
    (10.0, "W21X50"),
    (12.0, "W21X57"),
    (14.0, "W24X68"),
    (16.0, "W24X84"),
]

# Bracing section lookup: (max_bay_diagonal_m) → section
BRACING_SELECTION = [
    (5.0,  "L3X3X5/16"),
    (6.0,  "L3.5X3.5X5/16"),
    (7.0,  "L4X4X3/8"),
    (8.0,  "L4X4X1/2"),
    (10.0, "L5X5X3/8"),
    (12.0, "L5X5X1/2"),
    (15.0, "L6X6X1/2"),
]


def select_column_section(height_m: float, tributary_area_m2: float) -> str:
    """Select appropriate column section based on height and tributary area."""
    for max_h, max_a, section in COLUMN_SELECTION:
        if height_m <= max_h and tributary_area_m2 <= max_a:
            return section
    return "W14X159"  # Largest fallback


def select_beam_section(span_m: float) -> str:
    """Select appropriate beam section based on span."""
    for max_span, section in BEAM_SELECTION:
        if span_m <= max_span:
            return section
    return "W24X84"  # Largest fallback


def select_bracing_section(diagonal_m: float) -> str:
    """Select appropriate bracing section based on diagonal length."""
    for max_diag, section in BRACING_SELECTION:
        if diagonal_m <= max_diag:
            return section
    return "L6X6X1/2"


# ──────────────────────────────────────────────────────────────
# Platform sizing rules
# ──────────────────────────────────────────────────────────────

@dataclass
class PlatformSizing:
    """Derived platform dimensions from vessel/equipment parameters."""
    inner_width: float   # m
    outer_width: float   # m
    length: float        # m

    @staticmethod
    def from_vessel(vessel_od_mm: float, vessel_length_mm: float,
                    clearance_mm: float = 150, walkway_mm: float = 1200,
                    overhang_mm: float = 1000) -> "PlatformSizing":
        """Calculate platform dimensions from vessel geometry."""
        inner = (vessel_od_mm + 2 * clearance_mm) / 1000
        outer = inner + 2 * walkway_mm / 1000
        length = (vessel_length_mm + 2 * overhang_mm) / 1000
        return PlatformSizing(inner_width=inner, outer_width=outer, length=length)


# ──────────────────────────────────────────────────────────────
# Load derivation
# ──────────────────────────────────────────────────────────────

def area_load_to_udl(load_kpa: float, tributary_width_m: float) -> float:
    """Convert area load (kPa) to member UDL (kN/m) via tributary width."""
    return load_kpa * tributary_width_m


def max_beam_span_for_section(section_name: str, load_kn_m: float = 10.0) -> float:
    """Rough estimate of max span for a beam section under given UDL (serviceability)."""
    section = get_section(section_name)
    if section is None:
        return 6.0  # conservative default
    ix = section.get("Ix_mm4", 100e6)
    # Simplified: L = (384 * E * I / (5 * w * L/360))^(1/4)
    # Rough approximation based on deflection limit L/360
    e = 200_000  # MPa
    # w = load_kn_m * 1000 (N/m)
    w = load_kn_m  # kN/m → already in kN
    # Very rough: max span ≈ (ix / 1e6) ^ 0.4 * 4
    return min((ix / 1e6) ** 0.4 * 4, 20.0)
