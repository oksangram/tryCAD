"""
Steel material properties for structural design.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SteelGrade:
    name: str
    fy_mpa: float          # Yield strength
    fu_mpa: float          # Ultimate strength
    e_mpa: float           # Elastic modulus
    density_kg_m3: float   # Density
    poisson: float         # Poisson's ratio
    alpha: float           # Thermal expansion coefficient (per °C)


# ── Standard grades ──

A36 = SteelGrade(
    name="A36",
    fy_mpa=250, fu_mpa=400, e_mpa=200_000,
    density_kg_m3=7850, poisson=0.3, alpha=12e-6,
)

A572_GR50 = SteelGrade(
    name="A572GR50",
    fy_mpa=345, fu_mpa=450, e_mpa=200_000,
    density_kg_m3=7850, poisson=0.3, alpha=12e-6,
)

A992 = SteelGrade(
    name="A992",
    fy_mpa=345, fu_mpa=450, e_mpa=200_000,
    density_kg_m3=7850, poisson=0.3, alpha=12e-6,
)

GRADES = {"A36": A36, "A572GR50": A572_GR50, "A992": A992}


def get_grade(name: str) -> SteelGrade:
    """Get a steel grade by name. Falls back to A36 if not found."""
    return GRADES.get(name.upper().replace(" ", "").replace("-", ""), A36)
