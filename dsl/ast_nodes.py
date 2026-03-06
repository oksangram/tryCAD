"""
AST node definitions for the STAAD Pro-like DSL.

These dataclasses represent the parsed structure of a DSL script.
They are produced by the parser and consumed by the writer and validator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────

class StructureType(Enum):
    SPACE = "SPACE"         # 3D analysis
    PLANE = "PLANE"         # 2D frame analysis
    TRUSS = "TRUSS"         # Truss analysis (axial only)
    FLOOR = "FLOOR"         # Floor/slab analysis


class SupportType(Enum):
    FIXED = "FIXED"
    PINNED = "PINNED"
    FIXED_BUT = "FIXED BUT"  # Partial fixity (release specific DOFs)


class LoadType(Enum):
    UNI = "UNI"              # Uniform distributed load
    CON = "CON"              # Concentrated (point) load
    LIN = "LIN"              # Linearly varying load
    TRAP = "TRAP"            # Trapezoidal load


class LoadDirection(Enum):
    GX = "GX"    # Global X
    GY = "GY"    # Global Y (typically vertical)
    GZ = "GZ"    # Global Z
    X = "X"      # Local x-axis
    Y = "Y"      # Local y-axis
    Z = "Z"      # Local z-axis


class BracingType(Enum):
    X_BRACE = "X"
    K_BRACE = "K"
    V_BRACE = "V"
    CHEVRON = "CHEVRON"


# ──────────────────────────────────────────────────────────────
# Geometry Nodes
# ──────────────────────────────────────────────────────────────

@dataclass
class Joint:
    """A point in 3D space."""
    id: int
    x: float
    y: float
    z: float

    def coords(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: Joint) -> float:
        return ((self.x - other.x) ** 2 +
                (self.y - other.y) ** 2 +
                (self.z - other.z) ** 2) ** 0.5


@dataclass
class Member:
    """A structural member connecting two joints."""
    id: int
    start_joint: int   # Joint ID
    end_joint: int     # Joint ID


# ──────────────────────────────────────────────────────────────
# Property Nodes
# ──────────────────────────────────────────────────────────────

@dataclass
class SectionAssignment:
    """Assigns a steel section to one or more members."""
    member_ids: list[int]
    section_name: str        # e.g., "W12X65", "L4X4X3/8"
    spec_type: str = "ST"    # ST=standard, LD=double angle, SD=single angle


@dataclass
class MaterialAssignment:
    """Assigns material properties to members."""
    member_ids: list[int]
    material_name: str       # e.g., "STEEL", "A36"


# ──────────────────────────────────────────────────────────────
# Support Nodes
# ──────────────────────────────────────────────────────────────

@dataclass
class Support:
    """Boundary condition at a joint."""
    joint_ids: list[int]
    support_type: SupportType
    released_dofs: Optional[list[str]] = None  # For FIXED BUT, e.g. ["MZ"]


# ──────────────────────────────────────────────────────────────
# Load Nodes
# ──────────────────────────────────────────────────────────────

@dataclass
class MemberLoad:
    """A load applied to one or more members."""
    member_ids: list[int]
    load_type: LoadType
    direction: LoadDirection
    value: float                # Primary load value (kN/m for UNI, kN for CON)
    distance: Optional[float] = None  # Distance from start for CON loads


@dataclass
class JointLoad:
    """A load applied directly to a joint."""
    joint_id: int
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0


@dataclass
class SelfWeight:
    """Self-weight load."""
    direction: LoadDirection
    factor: float = -1.0  # Typically -1 for gravity in GY


@dataclass
class LoadCase:
    """A named load case containing multiple loads."""
    id: int
    title: str = ""
    member_loads: list[MemberLoad] = field(default_factory=list)
    joint_loads: list[JointLoad] = field(default_factory=list)
    self_weight: Optional[SelfWeight] = None


@dataclass
class LoadCombination:
    """A combination of load cases with factors."""
    id: int
    title: str = ""
    factors: dict[int, float] = field(default_factory=dict)  # load_case_id → factor


# ──────────────────────────────────────────────────────────────
# Top-Level Script Node
# ──────────────────────────────────────────────────────────────

@dataclass
class Script:
    """Root AST node representing a complete STAAD Pro script."""
    structure_type: StructureType = StructureType.SPACE
    input_width: int = 79
    unit_length: str = "METER"
    unit_force: str = "KN"

    joints: list[Joint] = field(default_factory=list)
    members: list[Member] = field(default_factory=list)
    section_assignments: list[SectionAssignment] = field(default_factory=list)
    material_assignments: list[MaterialAssignment] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)
    load_cases: list[LoadCase] = field(default_factory=list)
    load_combinations: list[LoadCombination] = field(default_factory=list)
    perform_analysis: bool = True

    # ── Convenience accessors ──

    def get_joint(self, joint_id: int) -> Optional[Joint]:
        """Get joint by ID. Returns None if not found."""
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None

    def get_member(self, member_id: int) -> Optional[Member]:
        """Get member by ID. Returns None if not found."""
        for m in self.members:
            if m.id == member_id:
                return m
        return None

    def joint_ids(self) -> set[int]:
        return {j.id for j in self.joints}

    def member_ids(self) -> set[int]:
        return {m.id for m in self.members}

    def max_joint_id(self) -> int:
        return max((j.id for j in self.joints), default=0)

    def max_member_id(self) -> int:
        return max((m.id for m in self.members), default=0)

    def bounding_box(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Returns ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
        if not self.joints:
            return ((0, 0, 0), (0, 0, 0))
        xs = [j.x for j in self.joints]
        ys = [j.y for j in self.joints]
        zs = [j.z for j in self.joints]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
