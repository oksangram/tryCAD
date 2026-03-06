"""
Beam grid tool — creates a regular grid of beams on a horizontal plane.
"""

from __future__ import annotations
from .session import ToolSession


def create_beam_grid(session: ToolSession, elevation: float,
                     x_positions: list[float], z_positions: list[float],
                     main_section: str,
                     secondary_x_positions: list[float] = None,
                     secondary_z_positions: list[float] = None,
                     secondary_section: str = None) -> str:
    """
    Create a grid of beams at a given elevation.

    Main beams run along grid lines. Optional secondary beams at finer spacing.

    Args:
        session: The active tool session.
        elevation: Y-coordinate (elevation) for the beam grid.
        x_positions: X-coordinates of main grid lines.
        z_positions: Z-coordinates of main grid lines.
        main_section: Section for main beams.
        secondary_x_positions: Optional, X-coords for secondary beams.
        secondary_z_positions: Optional, Z-coords for secondary beams.
        secondary_section: Section for secondary beams.

    Returns:
        Human-readable summary.
    """
    main_beam_ids = []
    sec_beam_ids = []

    # All X positions including secondary
    all_x = sorted(set(x_positions + (secondary_x_positions or [])))
    all_z = sorted(set(z_positions + (secondary_z_positions or [])))

    # Create joints at all grid intersections
    joint_grid = {}  # (x_idx, z_idx) → joint_id using actual coords
    for x in all_x:
        for z in all_z:
            jid = session.add_joint(x, elevation, z)
            joint_grid[(x, z)] = jid

    # Main beams along X-direction (at each main Z line)
    for z in z_positions:
        for i in range(len(all_x) - 1):
            x1, x2 = all_x[i], all_x[i + 1]
            j1 = joint_grid[(x1, z)]
            j2 = joint_grid[(x2, z)]
            mid = session.add_member(j1, j2)
            # Check if this is a main or secondary beam span
            if x1 in x_positions and x2 in x_positions:
                main_beam_ids.append(mid)
            elif all(x in x_positions for x in [x1, x2]):
                main_beam_ids.append(mid)
            else:
                sec_beam_ids.append(mid)

    # Main beams along Z-direction (at each main X line)
    for x in x_positions:
        for i in range(len(all_z) - 1):
            z1, z2 = all_z[i], all_z[i + 1]
            j1 = joint_grid[(x, z1)]
            j2 = joint_grid[(x, z2)]
            mid = session.add_member(j1, j2)
            if z1 in z_positions and z2 in z_positions:
                main_beam_ids.append(mid)
            else:
                sec_beam_ids.append(mid)

    # Secondary beams along Z-direction (at secondary X lines)
    if secondary_x_positions:
        sec_x_only = [x for x in secondary_x_positions if x not in x_positions]
        for x in sec_x_only:
            for i in range(len(all_z) - 1):
                z1, z2 = all_z[i], all_z[i + 1]
                if (x, z1) in joint_grid and (x, z2) in joint_grid:
                    j1 = joint_grid[(x, z1)]
                    j2 = joint_grid[(x, z2)]
                    mid = session.add_member(j1, j2)
                    sec_beam_ids.append(mid)

    # Secondary beams along X-direction (at secondary Z lines)
    if secondary_z_positions:
        sec_z_only = [z for z in secondary_z_positions if z not in z_positions]
        for z in sec_z_only:
            for i in range(len(all_x) - 1):
                x1, x2 = all_x[i], all_x[i + 1]
                if (x1, z) in joint_grid and (x2, z) in joint_grid:
                    j1 = joint_grid[(x1, z)]
                    j2 = joint_grid[(x2, z)]
                    mid = session.add_member(j1, j2)
                    sec_beam_ids.append(mid)

    # Assign sections
    if main_beam_ids:
        session.assign_section(main_beam_ids, main_section)
    if sec_beam_ids and secondary_section:
        session.assign_section(sec_beam_ids, secondary_section)
    elif sec_beam_ids:
        session.assign_section(sec_beam_ids, main_section)

    # Summary
    x_extent = f"{min(x_positions):.1f} to {max(x_positions):.1f}"
    z_extent = f"{min(z_positions):.1f} to {max(z_positions):.1f}"

    result = (
        f"Created beam grid at elevation {elevation:.1f}m.\n"
        f"Extents: X=[{x_extent}], Z=[{z_extent}]\n"
        f"Main beams: {len(main_beam_ids)} members ({main_section})"
    )
    if sec_beam_ids:
        result += f"\nSecondary beams: {len(sec_beam_ids)} members ({secondary_section or main_section})"
    result += f"\n{session.get_summary()}"

    return result


def create_beam_spans(session: ToolSession,
                      joint_pairs: list[tuple[int, int]],
                      section: str) -> str:
    """
    Create individual beam spans between specified joint pairs.

    Args:
        session: The active tool session.
        joint_pairs: List of (start_joint_id, end_joint_id) tuples.
        section: Section for all beams.

    Returns:
        Human-readable summary.
    """
    beam_ids = []
    for start, end in joint_pairs:
        mid = session.add_member(start, end)
        beam_ids.append(mid)

    session.assign_section(beam_ids, section)

    return (
        f"Created {len(beam_ids)} beam spans with section {section}.\n"
        f"Members: {beam_ids}"
    )
