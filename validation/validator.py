"""
Validation engine — 5-layer deterministic validation for STAAD Pro scripts.

Layers:
1. Syntax (handled by parser — not in this module)
2. Referential integrity (cross-references between blocks)
3. Geometry feasibility (physical correctness)
4. Structural sanity (stability & connectivity)
5. Physical reasonableness (engineering sanity checks)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from dsl.ast_nodes import Script, Joint, Member


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate(script: Script) -> ValidationResult:
    """Run all validation layers and return a combined result."""
    errors = []
    warnings = []

    errors += check_referential_integrity(script)
    errors += check_geometry(script)
    warnings += check_structural_sanity(script)
    warnings += check_physical_reasonableness(script)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────
# Layer 2: Referential Integrity
# ──────────────────────────────────────────────────────────────

def check_referential_integrity(script: Script) -> list[str]:
    """Check that all cross-references are valid."""
    errors = []
    joint_ids = script.joint_ids()
    member_ids = script.member_ids()

    # Members reference valid joints
    for m in script.members:
        if m.start_joint not in joint_ids:
            errors.append(f"Member {m.id}: start joint {m.start_joint} not defined")
        if m.end_joint not in joint_ids:
            errors.append(f"Member {m.id}: end joint {m.end_joint} not defined")

    # Supports reference valid joints
    for sup in script.supports:
        for jid in sup.joint_ids:
            if jid not in joint_ids:
                errors.append(f"Support references undefined joint {jid}")

    # Section assignments reference valid members
    for sa in script.section_assignments:
        for mid in sa.member_ids:
            if mid not in member_ids:
                errors.append(f"Section '{sa.section_name}' assigned to undefined member {mid}")

    # Load member references
    for lc in script.load_cases:
        for ml in lc.member_loads:
            for mid in ml.member_ids:
                if mid not in member_ids:
                    errors.append(f"Load case {lc.id}: load on undefined member {mid}")
        for jl in lc.joint_loads:
            if jl.joint_id not in joint_ids:
                errors.append(f"Load case {lc.id}: load on undefined joint {jl.joint_id}")

    return errors


# ──────────────────────────────────────────────────────────────
# Layer 3: Geometry Feasibility
# ──────────────────────────────────────────────────────────────

def check_geometry(script: Script) -> list[str]:
    """Check geometric feasibility."""
    errors = []

    # Zero-length members
    for m in script.members:
        j1 = script.get_joint(m.start_joint)
        j2 = script.get_joint(m.end_joint)
        if j1 and j2:
            length = j1.distance_to(j2)
            if length < 0.001:
                errors.append(f"Member {m.id}: zero length (joints {m.start_joint} and {m.end_joint} at same location)")

    # Self-referencing members
    for m in script.members:
        if m.start_joint == m.end_joint:
            errors.append(f"Member {m.id}: start and end joint are the same ({m.start_joint})")

    # Duplicate joints (same coordinates, different IDs)
    coords_seen = {}
    for j in script.joints:
        key = (round(j.x, 3), round(j.y, 3), round(j.z, 3))
        if key in coords_seen:
            errors.append(f"Duplicate joints: {coords_seen[key]} and {j.id} at ({j.x}, {j.y}, {j.z})")
        else:
            coords_seen[key] = j.id

    # Duplicate members (same connectivity)
    member_pairs = {}
    for m in script.members:
        pair = tuple(sorted([m.start_joint, m.end_joint]))
        if pair in member_pairs:
            errors.append(f"Duplicate members: {member_pairs[pair]} and {m.id} (joints {pair})")
        else:
            member_pairs[pair] = m.id

    return errors


# ──────────────────────────────────────────────────────────────
# Layer 4: Structural Sanity
# ──────────────────────────────────────────────────────────────

def check_structural_sanity(script: Script) -> list[str]:
    """Check basic structural stability (warnings, not hard errors)."""
    warnings = []

    # Must have at least one support
    if not script.supports:
        warnings.append("No supports defined — structure will be unstable")

    # Count reactions
    total_reactions = 0
    for sup in script.supports:
        n_joints = len(sup.joint_ids)
        if sup.support_type.value == "FIXED":
            total_reactions += n_joints * 6  # 3 forces + 3 moments per joint
        elif sup.support_type.value == "PINNED":
            total_reactions += n_joints * 3  # 3 forces per joint

    n_joints = len(script.joints)
    n_members = len(script.members)

    # 3D: m + r >= 3j for stability (necessary, not sufficient)
    if n_members + total_reactions < 3 * n_joints:
        warnings.append(
            f"Possible mechanism: m({n_members}) + r({total_reactions}) = "
            f"{n_members + total_reactions} < 3j = {3 * n_joints}"
        )

    # Check connectivity via BFS
    if script.members and script.joints:
        adjacency = {}
        for m in script.members:
            adjacency.setdefault(m.start_joint, set()).add(m.end_joint)
            adjacency.setdefault(m.end_joint, set()).add(m.start_joint)

        all_jids = script.joint_ids()
        start = next(iter(all_jids))
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        connected_joints = visited & all_jids
        if len(connected_joints) < len(all_jids):
            disconnected = all_jids - connected_joints
            warnings.append(f"Disconnected structure: joints {disconnected} not connected to main frame")

    return warnings


# ──────────────────────────────────────────────────────────────
# Layer 5: Physical Reasonableness
# ──────────────────────────────────────────────────────────────

def check_physical_reasonableness(script: Script) -> list[str]:
    """Check for physically unreasonable values."""
    warnings = []

    # Member lengths
    for m in script.members:
        j1 = script.get_joint(m.start_joint)
        j2 = script.get_joint(m.end_joint)
        if j1 and j2:
            length = j1.distance_to(j2)
            if length > 30.0:
                warnings.append(f"Member {m.id}: very long ({length:.1f}m)")
            elif length < 0.3:
                warnings.append(f"Member {m.id}: very short ({length:.3f}m)")

    # Load magnitudes
    for lc in script.load_cases:
        for ml in lc.member_loads:
            if abs(ml.value) > 500:
                warnings.append(f"Load case {lc.id}: very high load {ml.value} kN/m on members {ml.member_ids}")
        for jl in lc.joint_loads:
            total = abs(jl.fx) + abs(jl.fy) + abs(jl.fz)
            if total > 5000:
                warnings.append(f"Load case {lc.id}: very high joint load ({total:.0f} kN) at joint {jl.joint_id}")

    # Orphan joints (defined but not in any member)
    used_joints = set()
    for m in script.members:
        used_joints.add(m.start_joint)
        used_joints.add(m.end_joint)

    orphans = script.joint_ids() - used_joints
    if orphans:
        warnings.append(f"Orphan joints (not in any member): {orphans}")

    return warnings
