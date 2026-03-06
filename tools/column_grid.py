"""
Column grid tool — creates vertical columns on a regular grid.
"""

from __future__ import annotations
from .session import ToolSession


def create_column_grid(session: ToolSession, grid_x: list[float], grid_z: list[float],
                       base_y: float, top_y: float, section: str) -> str:
    """
    Create vertical columns at every intersection of grid_x and grid_z lines.

    Args:
        session: The active tool session.
        grid_x: X-coordinates of column lines (meters).
        grid_z: Z-coordinates of column lines (meters).
        base_y: Base elevation (meters).
        top_y: Top elevation (meters).
        section: AISC section designation (e.g., "W12X65").

    Returns:
        Human-readable summary of what was created.
    """
    base_joints = []
    top_joints = []
    column_member_ids = []

    for z in grid_z:
        for x in grid_x:
            base_id = session.add_joint(x, base_y, z)
            top_id = session.add_joint(x, top_y, z)
            base_joints.append(base_id)
            top_joints.append(top_id)
            mid = session.add_member(base_id, top_id)
            column_member_ids.append(mid)

    session.assign_section(column_member_ids, section)

    n_cols = len(column_member_ids)
    height = top_y - base_y
    n_x = len(grid_x)
    n_z = len(grid_z)

    # Build summary
    base_list = ", ".join(f"{jid}({session.get_joint(jid).x:.1f}, {session.get_joint(jid).y:.1f}, {session.get_joint(jid).z:.1f})"
                          for jid in base_joints)
    top_list = ", ".join(f"{jid}({session.get_joint(jid).x:.1f}, {session.get_joint(jid).y:.1f}, {session.get_joint(jid).z:.1f})"
                         for jid in top_joints)

    return (
        f"Created {n_cols} columns on a {n_x}x{n_z} grid. "
        f"Height: {height:.1f}m. Section: {section}.\n"
        f"Base joints: {base_list}\n"
        f"Top joints: {top_list}\n"
        f"Column members: {column_member_ids}"
    )
