"""
STAAD Pro DSL Writer.

Serializes a Script AST back into well-formatted DSL text.
"""

from __future__ import annotations
from .ast_nodes import (
    Script, Joint, Member, SectionAssignment, Support, SupportType,
    MemberLoad, JointLoad, SelfWeight, LoadCase, LoadCombination,
    LoadType, LoadDirection, StructureType,
)


def write(script: Script) -> str:
    """
    Serialize a Script AST into STAAD Pro DSL text.

    Args:
        script: The Script AST to serialize.

    Returns:
        str: Formatted DSL text.
    """
    lines: list[str] = []

    # ── Header ──
    lines.append(f"STAAD {script.structure_type.value}")
    lines.append(f"INPUT WIDTH {script.input_width}")
    lines.append(f"UNIT {script.unit_length} {script.unit_force}")
    lines.append("")

    # ── Joint Coordinates ──
    if script.joints:
        lines.append("JOINT COORDINATES")
        for j in sorted(script.joints, key=lambda j: j.id):
            lines.append(f"{j.id} {j.x:.3f} {j.y:.3f} {j.z:.3f}")
        lines.append("")

    # ── Member Incidences ──
    if script.members:
        lines.append("MEMBER INCIDENCES")
        for m in sorted(script.members, key=lambda m: m.id):
            lines.append(f"{m.id} {m.start_joint} {m.end_joint}")
        lines.append("")

    # ── Member Properties ──
    if script.section_assignments:
        lines.append("MEMBER PROPERTY")
        for sa in script.section_assignments:
            ids_str = " ".join(str(i) for i in sa.member_ids)
            lines.append(f"{ids_str} TABLE {sa.spec_type} {sa.section_name}")
        lines.append("")

    # ── Supports ──
    if script.supports:
        lines.append("SUPPORT")
        for sup in script.supports:
            ids_str = " ".join(str(i) for i in sup.joint_ids)
            line = f"{ids_str} {sup.support_type.value}"
            if sup.support_type == SupportType.FIXED_BUT and sup.released_dofs:
                line = f"{ids_str} FIXED BUT " + " ".join(sup.released_dofs)
            lines.append(line)
        lines.append("")

    # ── Loading ──
    for lc in script.load_cases:
        title_part = f" {lc.title}" if lc.title else ""
        lines.append(f"LOADING {lc.id}{title_part}")

        if lc.self_weight:
            sw = lc.self_weight
            lines.append(f"SELFWEIGHT {sw.direction.value} {sw.factor}")

        if lc.member_loads:
            lines.append("MEMBER LOAD")
            for ml in lc.member_loads:
                ids_str = " ".join(str(i) for i in ml.member_ids)
                line = f"{ids_str} {ml.load_type.value} {ml.direction.value} {ml.value}"
                if ml.distance is not None:
                    line += f" {ml.distance}"
                lines.append(line)

        if lc.joint_loads:
            lines.append("JOINT LOAD")
            for jl in lc.joint_loads:
                line = f"{jl.joint_id} FX {jl.fx} FY {jl.fy} FZ {jl.fz}"
                if jl.mx != 0 or jl.my != 0 or jl.mz != 0:
                    line += f" MX {jl.mx} MY {jl.my} MZ {jl.mz}"
                lines.append(line)

        lines.append("")

    # ── Load Combinations ──
    for lc in script.load_combinations:
        title_part = f" {lc.title}" if lc.title else ""
        lines.append(f"LOAD COMBINATION {lc.id}{title_part}")
        for case_id, factor in sorted(lc.factors.items()):
            lines.append(f"{case_id} {factor}")
        lines.append("")

    # ── Analysis ──
    if script.perform_analysis:
        lines.append("PERFORM ANALYSIS")
        lines.append("")

    # ── Footer ──
    lines.append("FINISH")
    lines.append("")

    return "\n".join(lines)


def write_compact(script: Script) -> str:
    """
    Serialize with semicolons to reduce line count (STAAD format variant).
    Multiple joints/members per line, separated by semicolons.
    """
    lines: list[str] = []

    # ── Header ──
    lines.append(f"STAAD {script.structure_type.value}")
    lines.append(f"INPUT WIDTH 79")
    lines.append(f"UNIT {script.unit_length} {script.unit_force}")

    # ── Joint Coordinates (compact: multiple per line) ──
    if script.joints:
        lines.append("JOINT COORDINATES")
        sorted_joints = sorted(script.joints, key=lambda j: j.id)
        chunk_size = 3  # 3 joints per line
        for i in range(0, len(sorted_joints), chunk_size):
            chunk = sorted_joints[i:i + chunk_size]
            parts = [f"{j.id} {j.x:.3f} {j.y:.3f} {j.z:.3f}" for j in chunk]
            lines.append(" ; ".join(parts))

    # ── Member Incidences (compact) ──
    if script.members:
        lines.append("MEMBER INCIDENCES")
        sorted_members = sorted(script.members, key=lambda m: m.id)
        chunk_size = 4
        for i in range(0, len(sorted_members), chunk_size):
            chunk = sorted_members[i:i + chunk_size]
            parts = [f"{m.id} {m.start_joint} {m.end_joint}" for m in chunk]
            lines.append(" ; ".join(parts))

    # Properties, supports, loads — same as regular write
    if script.section_assignments:
        lines.append("MEMBER PROPERTY")
        for sa in script.section_assignments:
            ids_str = " ".join(str(i) for i in sa.member_ids)
            lines.append(f"{ids_str} TABLE {sa.spec_type} {sa.section_name}")

    if script.supports:
        lines.append("SUPPORT")
        for sup in script.supports:
            ids_str = " ".join(str(i) for i in sup.joint_ids)
            lines.append(f"{ids_str} {sup.support_type.value}")

    for lc in script.load_cases:
        title_part = f" {lc.title}" if lc.title else ""
        lines.append(f"LOADING {lc.id}{title_part}")
        if lc.self_weight:
            lines.append(f"SELFWEIGHT {lc.self_weight.direction.value} {lc.self_weight.factor}")
        if lc.member_loads:
            lines.append("MEMBER LOAD")
            for ml in lc.member_loads:
                ids_str = " ".join(str(i) for i in ml.member_ids)
                lines.append(f"{ids_str} {ml.load_type.value} {ml.direction.value} {ml.value}")
        if lc.joint_loads:
            lines.append("JOINT LOAD")
            for jl in lc.joint_loads:
                lines.append(f"{jl.joint_id} FX {jl.fx} FY {jl.fy} FZ {jl.fz}")

    for lc in script.load_combinations:
        title_part = f" {lc.title}" if lc.title else ""
        lines.append(f"LOAD COMBINATION {lc.id}{title_part}")
        for case_id, factor in sorted(lc.factors.items()):
            lines.append(f"{case_id} {factor}")

    if script.perform_analysis:
        lines.append("PERFORM ANALYSIS")

    lines.append("FINISH")
    return "\n".join(lines)
