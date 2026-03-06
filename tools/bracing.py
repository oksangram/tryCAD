"""
Bracing tool — adds X/K/V/Chevron bracing in a vertical bay.
"""

from __future__ import annotations
import math
from .session import ToolSession


def add_bracing(session: ToolSession, bay_corners: list[list[int]],
                brace_type: str = "X", section: str = "L4X4X3/8") -> str:
    """
    Add bracing in one or more bays defined by corner joints.

    Args:
        session: The active tool session.
        bay_corners: List of bays, each bay is [bottom_left, bottom_right, top_left, top_right] joint IDs.
        brace_type: "X", "K", "V", or "CHEVRON".
        section: Bracing section designation.

    Returns:
        Human-readable summary.
    """
    brace_member_ids = []

    for corners in bay_corners:
        bl, br, tl, tr = corners  # bottom-left, bottom-right, top-left, top-right

        if brace_type == "X":
            # Two diagonals: BL→TR and BR→TL
            mid1 = session.add_member(bl, tr)
            mid2 = session.add_member(br, tl)
            brace_member_ids.extend([mid1, mid2])

        elif brace_type == "V":
            # V-brace: two diagonals meeting at midpoint of bottom beam
            j_bl = session.get_joint(bl)
            j_br = session.get_joint(br)
            mid_x = (j_bl.x + j_br.x) / 2
            mid_y = j_bl.y  # bottom elevation
            mid_z = (j_bl.z + j_br.z) / 2
            mid_jid = session.add_joint(mid_x, mid_y, mid_z)
            mid1 = session.add_member(mid_jid, tl)
            mid2 = session.add_member(mid_jid, tr)
            brace_member_ids.extend([mid1, mid2])

        elif brace_type == "CHEVRON":
            # Inverted-V: diagonals meeting at midpoint of top beam
            j_tl = session.get_joint(tl)
            j_tr = session.get_joint(tr)
            mid_x = (j_tl.x + j_tr.x) / 2
            mid_y = j_tl.y  # top elevation
            mid_z = (j_tl.z + j_tr.z) / 2
            mid_jid = session.add_joint(mid_x, mid_y, mid_z)
            mid1 = session.add_member(bl, mid_jid)
            mid2 = session.add_member(br, mid_jid)
            brace_member_ids.extend([mid1, mid2])

        elif brace_type == "K":
            # K-brace: diagonals meeting at midpoint of one column
            j_bl = session.get_joint(bl)
            j_tl = session.get_joint(tl)
            mid_x = j_bl.x
            mid_y = (j_bl.y + j_tl.y) / 2
            mid_z = j_bl.z
            mid_jid = session.add_joint(mid_x, mid_y, mid_z)
            mid1 = session.add_member(mid_jid, br)
            mid2 = session.add_member(mid_jid, tr)
            brace_member_ids.extend([mid1, mid2])

    # Assign section (double angle for bracing)
    if brace_member_ids:
        spec = "LD" if section.startswith("L") else "ST"
        session.assign_section(brace_member_ids, section, spec_type=spec)

    return (
        f"Created {brace_type}-bracing in {len(bay_corners)} bay(s). "
        f"{len(brace_member_ids)} brace members ({section}).\n"
        f"Brace member IDs: {brace_member_ids}"
    )
