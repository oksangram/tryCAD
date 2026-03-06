"""
Support and load tools for structural design.
"""

from __future__ import annotations
from .session import ToolSession


def add_supports(session: ToolSession, joints: list[int],
                 support_type: str = "FIXED") -> str:
    """
    Add boundary condition supports at specified joints.

    Args:
        session: The active tool session.
        joints: List of joint IDs to support.
        support_type: "FIXED" or "PINNED".

    Returns:
        Human-readable summary.
    """
    session.add_supports(joints, support_type)

    joint_descriptions = []
    for jid in joints:
        j = session.get_joint(jid)
        if j:
            joint_descriptions.append(f"{jid}({j.x:.1f}, {j.y:.1f}, {j.z:.1f})")

    return (
        f"Added {support_type} supports at {len(joints)} joints.\n"
        f"Supported joints: {', '.join(joint_descriptions)}"
    )


def add_member_load(session: ToolSession, members: list[int],
                    load_type: str, direction: str, value: float,
                    case_id: int = 1, case_title: str = "") -> str:
    """
    Add a distributed or concentrated load to members.

    Args:
        session: The active tool session.
        members: List of member IDs to load.
        load_type: "UNI", "CON", "LIN", or "TRAP".
        direction: "GX", "GY", "GZ", "X", "Y", or "Z".
        value: Load magnitude (kN/m for UNI, kN for CON). Negative = downward for GY.
        case_id: Load case ID (default 1).
        case_title: Optional load case title.

    Returns:
        Human-readable summary.
    """
    session.add_member_load(case_id, members, load_type, direction, value)

    return (
        f"Added {load_type} load of {value} kN/m in {direction} direction "
        f"to {len(members)} members (members {members}).\n"
        f"Load case: {case_id}" + (f" ({case_title})" if case_title else "")
    )


def place_equipment(session: ToolSession, weight_kn: float,
                    x: float, z: float, elevation: float,
                    case_id: int = 1) -> str:
    """
    Place an equipment point load at specified location.
    Distributes to the nearest joint at the given elevation.

    Args:
        session: The active tool session.
        weight_kn: Equipment weight in kN.
        x: X-coordinate of equipment center.
        z: Z-coordinate of equipment center.
        elevation: Y-elevation of the platform.
        case_id: Load case ID.

    Returns:
        Human-readable summary.
    """
    # Find or create a joint at the equipment location
    equip_jid = session.add_joint(x, elevation, z)

    # Apply as a joint load (downward = negative GY)
    session.add_joint_load(case_id, equip_jid, fy=-weight_kn)

    return (
        f"Placed equipment load of {weight_kn} kN at "
        f"joint {equip_jid} ({x:.1f}, {elevation:.1f}, {z:.1f}).\n"
        f"Applied as point load FY={-weight_kn} kN in load case {case_id}."
    )
