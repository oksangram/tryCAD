"""
Tool-call sequence generator.

Takes structure parameters and generates:
1. Executes tools to build the structure (capturing all calls and results)
2. Creates a validated STAAD script
3. Returns the full tool-call trace for training data

This is the core engine — it deterministically produces training examples.
"""

from __future__ import annotations
import json
import math
from typing import Optional

from tools.session import ToolSession
from tools.column_grid import create_column_grid
from tools.beam_grid import create_beam_grid, create_beam_spans
from tools.bracing import add_bracing
from tools.loads import add_supports, add_member_load, place_equipment
from validation.validator import validate
from dsl.writer import write

from .sampler import StructureParams



def generate_portal_frame(params: StructureParams) -> Optional[dict]:
    """
    Generate a portal frame using tool calls.
    Returns a dict with the full tool-call trace, or None if validation fails.
    """
    session = ToolSession()
    calls = []

    # Grid positions
    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]

    for level in range(params.n_levels):
        base_y = params.base_elevation + level * params.height_per_level
        top_y = base_y + params.height_per_level

        # Step 1: Columns
        args = {
            "grid_x": grid_x,
            "grid_z": [0.0],
            "base_y": base_y,
            "top_y": top_y,
            "section": params.column_section,
        }
        result = create_column_grid(session, **args)
        calls.append({"tool": "create_column_grid", "args": args, "result": result})

    # Step 2: Beams at each level top
    for level in range(params.n_levels):
        elev = params.base_elevation + (level + 1) * params.height_per_level
        top_joints = session.find_joints_at_elevation(elev)
        # Sort by X coordinate
        top_joints_sorted = sorted(top_joints,
                                   key=lambda jid: session.get_joint(jid).x)

        if len(top_joints_sorted) >= 2:
            pairs = [(top_joints_sorted[i], top_joints_sorted[i + 1])
                     for i in range(len(top_joints_sorted) - 1)]
            args = {"joint_pairs": pairs, "section": params.beam_section_x}
            result = create_beam_spans(session, **args)
            calls.append({"tool": "create_beam_spans", "args": args, "result": result})

    # Step 3: Bracing (only at first level for portal frames)
    if params.bracing_type != "NONE" and params.bracing_bays:
        base_y = params.base_elevation
        top_y = base_y + params.height_per_level

        base_joints = sorted(
            session.find_joints_at_elevation(base_y),
            key=lambda jid: session.get_joint(jid).x
        )
        top_joints = sorted(
            session.find_joints_at_elevation(top_y),
            key=lambda jid: session.get_joint(jid).x
        )

        bay_corners = []
        for bay_idx in params.bracing_bays:
            if bay_idx < len(base_joints) - 1:
                bl = base_joints[bay_idx]
                br = base_joints[bay_idx + 1]
                tl = top_joints[bay_idx]
                tr = top_joints[bay_idx + 1]
                bay_corners.append([bl, br, tl, tr])

        if bay_corners:
            args = {
                "bay_corners": bay_corners,
                "brace_type": params.bracing_type,
                "section": params.bracing_section,
            }
            result = add_bracing(session, **args)
            calls.append({"tool": "add_bracing", "args": args, "result": result})

    # Step 4: Supports
    support_joints = sorted(
        session.find_joints_at_elevation(params.base_elevation),
        key=lambda jid: session.get_joint(jid).x
    )
    args = {"joints": support_joints, "support_type": params.support_type}
    result = add_supports(session, **args)
    calls.append({"tool": "add_supports", "args": args, "result": result})

    # Step 5: Loads
    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.tributary_width
    # Find beam members at the top level
    top_elev = params.base_elevation + params.n_levels * params.height_per_level
    top_joints = session.find_joints_at_elevation(top_elev)
    beam_members = []
    for mid, m in session._members.items():
        j1 = session.get_joint(m.start_joint)
        j2 = session.get_joint(m.end_joint)
        if j1 and j2:
            # Horizontal member at top elevation
            if (abs(j1.y - top_elev) < 0.01 and abs(j2.y - top_elev) < 0.01
                    and abs(j1.y - j2.y) < 0.01):
                beam_members.append(mid)

    if beam_members:
        args = {
            "members": sorted(beam_members),
            "load_type": "UNI",
            "direction": "GY",
            "value": round(udl, 2),
            "case_id": 1,
        }
        result = add_member_load(session, **args)
        calls.append({"tool": "add_member_load", "args": args, "result": result})

    # Validate
    script = session.to_script()
    val_result = validate(script)

    if not val_result.is_valid:
        return None  # Skip invalid examples

    dsl_text = session.to_dsl()

    return {
        "params": _params_to_dict(params),
        "tool_calls": calls,
        "dsl_script": dsl_text,
        "validation": {
            "is_valid": val_result.is_valid,
            "errors": val_result.errors,
            "warnings": val_result.warnings,
        },
        "stats": {
            "n_joints": len(session._joints),
            "n_members": len(session._members),
        },
    }


def generate_platform(params: StructureParams) -> Optional[dict]:
    """Generate a 3D platform using tool calls."""
    session = ToolSession()
    calls = []

    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]
    grid_z = [i * params.span_z for i in range(params.n_bays_z + 1)]

    for level in range(params.n_levels):
        base_y = params.base_elevation + level * params.height_per_level
        top_y = base_y + params.height_per_level

        args = {
            "grid_x": grid_x,
            "grid_z": grid_z,
            "base_y": base_y,
            "top_y": top_y,
            "section": params.column_section,
        }
        result = create_column_grid(session, **args)
        calls.append({"tool": "create_column_grid", "args": args, "result": result})

    # Beam grids at each level
    for level in range(params.n_levels):
        elev = params.base_elevation + (level + 1) * params.height_per_level
        args = {
            "elevation": elev,
            "x_positions": grid_x,
            "z_positions": grid_z,
            "main_section": params.beam_section_x,
        }
        result = create_beam_grid(session, **args)
        calls.append({"tool": "create_beam_grid", "args": args, "result": result})

    # Bracing (in XY plane at Z=0 and Z=max, for selected bays)
    if params.bracing_type != "NONE" and params.bracing_bays:
        base_y = params.base_elevation
        top_y = base_y + params.height_per_level

        for z_val in [grid_z[0], grid_z[-1]]:
            base_js = sorted(
                [jid for jid in session.find_joints_at_elevation(base_y)
                 if abs(session.get_joint(jid).z - z_val) < 0.01],
                key=lambda jid: session.get_joint(jid).x
            )
            top_js = sorted(
                [jid for jid in session.find_joints_at_elevation(top_y)
                 if abs(session.get_joint(jid).z - z_val) < 0.01],
                key=lambda jid: session.get_joint(jid).x
            )

            bay_corners = []
            for bay_idx in params.bracing_bays:
                if bay_idx < len(base_js) - 1 and bay_idx < len(top_js) - 1:
                    bay_corners.append([
                        base_js[bay_idx], base_js[bay_idx + 1],
                        top_js[bay_idx], top_js[bay_idx + 1],
                    ])

            if bay_corners:
                args = {
                    "bay_corners": bay_corners,
                    "brace_type": params.bracing_type,
                    "section": params.bracing_section,
                }
                result = add_bracing(session, **args)
                calls.append({"tool": "add_bracing", "args": args, "result": result})

    # Supports
    support_joints = sorted(session.find_joints_at_elevation(params.base_elevation))
    args = {"joints": support_joints, "support_type": params.support_type}
    result = add_supports(session, **args)
    calls.append({"tool": "add_supports", "args": args, "result": result})

    # Loads
    top_elev = params.base_elevation + params.n_levels * params.height_per_level
    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.span_z
    beam_members = _find_horizontal_members_at(session, top_elev)
    if beam_members:
        args = {
            "members": sorted(beam_members),
            "load_type": "UNI",
            "direction": "GY",
            "value": round(udl, 2),
            "case_id": 1,
        }
        result = add_member_load(session, **args)
        calls.append({"tool": "add_member_load", "args": args, "result": result})

    # Equipment loads
    for equip in params.equipment_loads:
        args = {
            "weight_kn": equip["weight_kn"],
            "x": equip["x"],
            "z": equip["z"],
            "elevation": top_elev,
            "case_id": 1,
        }
        result = place_equipment(session, **args)
        calls.append({"tool": "place_equipment", "args": args, "result": result})

    # Validate
    script = session.to_script()
    val_result = validate(script)
    if not val_result.is_valid:
        return None

    return {
        "params": _params_to_dict(params),
        "tool_calls": calls,
        "dsl_script": session.to_dsl(),
        "validation": {
            "is_valid": val_result.is_valid,
            "errors": val_result.errors,
            "warnings": val_result.warnings,
        },
        "stats": {
            "n_joints": len(session._joints),
            "n_members": len(session._members),
        },
    }


def generate_pipe_rack(params: StructureParams) -> Optional[dict]:
    """Generate a pipe rack using tool calls."""
    # Pipe racks are similar to platforms but elongated
    return generate_platform(params)


# ── Dispatcher ──

GENERATORS = {
    "portal_frame": generate_portal_frame,
    "platform": generate_platform,
    "pipe_rack": generate_pipe_rack,
}


def generate_structure(params: StructureParams) -> Optional[dict]:
    """Generate a structure from parameters using the appropriate generator."""
    gen = GENERATORS.get(params.structure_type)
    if gen is None:
        return None
    return gen(params)


# ── Helpers ──

def _find_horizontal_members_at(session: ToolSession, elevation: float) -> list[int]:
    """Find all horizontal members at a given elevation."""
    members = []
    for mid, m in session._members.items():
        j1 = session.get_joint(m.start_joint)
        j2 = session.get_joint(m.end_joint)
        if j1 and j2:
            if (abs(j1.y - elevation) < 0.01 and abs(j2.y - elevation) < 0.01
                    and abs(j1.y - j2.y) < 0.01):
                members.append(mid)
    return members


def _params_to_dict(params: StructureParams) -> dict:
    """Convert params to a JSON-serializable dict."""
    return {
        "structure_type": params.structure_type,
        "n_bays_x": params.n_bays_x,
        "n_bays_z": params.n_bays_z,
        "n_levels": params.n_levels,
        "span_x": params.span_x,
        "span_z": params.span_z,
        "height_per_level": params.height_per_level,
        "column_section": params.column_section,
        "beam_section_x": params.beam_section_x,
        "beam_section_z": params.beam_section_z,
        "bracing_type": params.bracing_type,
        "bracing_bays": params.bracing_bays,
        "bracing_section": params.bracing_section,
        "support_type": params.support_type,
        "load_dl_kpa": params.load_dl_kpa,
        "load_ll_kpa": params.load_ll_kpa,
        "tributary_width": params.tributary_width,
        "equipment_loads": params.equipment_loads,
        "title": params.title,
    }
