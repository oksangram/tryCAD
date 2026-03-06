"""
Modification task generator.

Creates training examples where the model must modify an existing structure:
- Add a level
- Add bracing to existing bays
- Change a section
- Extend with additional bays

These teach incremental design — working with an existing session state.
"""

from __future__ import annotations
import random
import json
from typing import Optional

from tools.session import ToolSession
from tools.column_grid import create_column_grid
from tools.beam_grid import create_beam_grid, create_beam_spans
from tools.bracing import add_bracing
from tools.loads import add_supports, add_member_load
from validation.validator import validate

from .sampler import sample_random_structure, StructureParams
from .nl_generator import generate_nl_spec


# Modification types
MODIFICATION_TYPES = [
    "add_level",
    "add_bracing",
    "extend_bays",
]


def generate_modification_example(params: Optional[StructureParams] = None) -> Optional[dict]:
    """Generate a modification training example."""
    if params is None:
        params = sample_random_structure()

    mod_type = random.choice(MODIFICATION_TYPES)

    if mod_type == "add_level":
        return _add_level_example(params)
    elif mod_type == "add_bracing":
        return _add_bracing_example(params)
    elif mod_type == "extend_bays":
        return _extend_bays_example(params)

    return None


def _add_level_example(params: StructureParams) -> Optional[dict]:
    """
    Build a single-level frame, then add a second level on top.
    """
    session = ToolSession()

    # --- Phase 1: Build initial 1-level frame ---
    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]
    h = params.height_per_level

    # Columns
    create_column_grid(session,
        grid_x=grid_x, grid_z=[0.0], base_y=0.0, top_y=h,
        section=params.column_section)

    # Beams
    top_joints = sorted(
        session.find_joints_at_elevation(h),
        key=lambda jid: session.get_joint(jid).x)
    if len(top_joints) >= 2:
        pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
        create_beam_spans(session, joint_pairs=pairs, section=params.beam_section_x)

    # Supports
    base_joints = sorted(session.find_joints_at_elevation(0.0))
    add_supports(session, joints=base_joints, support_type=params.support_type)

    initial_dsl = session.to_dsl()
    initial_joints = len(session._joints)
    initial_members = len(session._members)

    # --- Phase 2: User asks to add a level ---
    new_top = h * 2

    # New columns from h to 2h
    col_args = {
        "grid_x": grid_x, "grid_z": [0.0],
        "base_y": h, "top_y": new_top,
        "section": params.column_section,
    }
    col_result = create_column_grid(session, **col_args)

    # New beams at 2h
    new_top_joints = sorted(
        session.find_joints_at_elevation(new_top),
        key=lambda jid: session.get_joint(jid).x)
    if len(new_top_joints) < 2:
        return None

    new_pairs = [(new_top_joints[i], new_top_joints[i + 1]) for i in range(len(new_top_joints) - 1)]
    beam_args = {"joint_pairs": new_pairs, "section": params.beam_section_x}
    beam_result = create_beam_spans(session, **beam_args)

    # Add loads on the new level
    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.tributary_width
    new_beams = []
    for mid, m in session._members.items():
        j1 = session.get_joint(m.start_joint)
        j2 = session.get_joint(m.end_joint)
        if j1 and j2 and abs(j1.y - new_top) < 0.01 and abs(j2.y - new_top) < 0.01:
            new_beams.append(mid)

    if not new_beams:
        return None

    load_args = {
        "members": sorted(new_beams),
        "load_type": "UNI", "direction": "GY",
        "value": round(udl, 2), "case_id": 1,
    }
    load_result = add_member_load(session, **load_args)

    script = session.to_script()
    val_result = validate(script)
    if not val_result.is_valid:
        return None

    # Build messages
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"I have an existing {params.n_bays_x}-bay portal frame, {h}m high, "
            f"with {params.column_section} columns and {params.beam_section_x} beams. "
            f"Please add another level ({h}m) on top."
        )},
        {"role": "assistant", "content": (
            f"<think>\n"
            f"Adding a second level to the existing frame.\n"
            f"Current: {initial_joints} joints, {initial_members} members, height={h}m\n"
            f"Need: columns from {h}m to {new_top}m, new beams at {new_top}m\n"
            f"The existing top joints become the base of the new columns.\n"
            f"</think>"
        ), "tool_calls": [_fmt_call("call_1", "create_column_grid", col_args)]},
        {"role": "tool", "tool_call_id": "call_1", "content": col_result},
        {"role": "assistant", "content": "Columns for level 2 created. Now adding beams.",
         "tool_calls": [_fmt_call("call_2", "create_beam_spans", beam_args)]},
        {"role": "tool", "tool_call_id": "call_2", "content": beam_result},
        {"role": "assistant", "content": "Beams added. Now applying loads to the new level.",
         "tool_calls": [_fmt_call("call_3", "add_member_load", load_args)]},
        {"role": "tool", "tool_call_id": "call_3", "content": load_result},
        {"role": "assistant", "content": (
            f"Level 2 added successfully.\n\n"
            f"```\n{session.to_dsl()}\n```\n\n"
            f"Summary: {len(session._joints)} joints, {len(session._members)} members. "
            f"Total height now {new_top}m."
        )},
    ]

    return {"messages": messages, "category": 6}


def _add_bracing_example(params: StructureParams) -> Optional[dict]:
    """Build a frame without bracing, then add bracing at user's request."""
    session = ToolSession()

    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]

    create_column_grid(session,
        grid_x=grid_x, grid_z=[0.0], base_y=0.0,
        top_y=params.height_per_level, section=params.column_section)

    top_joints = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x)
    if len(top_joints) >= 2:
        pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
        create_beam_spans(session, joint_pairs=pairs, section=params.beam_section_x)

    base_joints = sorted(session.find_joints_at_elevation(0.0),
                         key=lambda jid: session.get_joint(jid).x)
    add_supports(session, joints=base_joints, support_type=params.support_type)

    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.tributary_width
    beam_members = []
    for mid, m in session._members.items():
        j1 = session.get_joint(m.start_joint)
        j2 = session.get_joint(m.end_joint)
        if j1 and j2 and abs(j1.y - params.height_per_level) < 0.01 and abs(j2.y - params.height_per_level) < 0.01:
            beam_members.append(mid)

    if beam_members:
        add_member_load(session,
            members=sorted(beam_members), load_type="UNI",
            direction="GY", value=round(udl, 2), case_id=1)

    if params.n_bays_x < 2:
        return None

    # Now add bracing at user's request
    brace_type = random.choice(["X", "K", "V", "CHEVRON"])
    brace_section = params.bracing_section or "L4X4X3/8"

    base_sorted = sorted(
        session.find_joints_at_elevation(0.0),
        key=lambda jid: session.get_joint(jid).x)
    top_sorted = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x)

    # Brace end bays
    bay_corners = []
    for bay_idx in [0, params.n_bays_x - 1]:
        if bay_idx < len(base_sorted) - 1:
            bay_corners.append([
                base_sorted[bay_idx], base_sorted[bay_idx + 1],
                top_sorted[bay_idx], top_sorted[bay_idx + 1],
            ])

    if not bay_corners:
        return None

    brace_args = {
        "bay_corners": bay_corners,
        "brace_type": brace_type,
        "section": brace_section,
    }
    brace_result = add_bracing(session, **brace_args)

    script = session.to_script()
    val_result = validate(script)
    if not val_result.is_valid:
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"The existing {params.n_bays_x}-bay portal frame needs lateral bracing. "
            f"Please add {brace_type}-bracing in the end bays using {brace_section} sections."
        )},
        {"role": "assistant", "content": (
            f"<think>\n"
            f"Adding {brace_type}-bracing to end bays (bay 1 and bay {params.n_bays_x}).\n"
            f"Need to identify the corner joints of each end bay.\n"
            f"Base joints at Y=0, top joints at Y={params.height_per_level}.\n"
            f"</think>"
        ), "tool_calls": [_fmt_call("call_1", "add_bracing", brace_args)]},
        {"role": "tool", "tool_call_id": "call_1", "content": brace_result},
        {"role": "assistant", "content": (
            f"Bracing added. {brace_type}-bracing in {len(bay_corners)} end bay(s).\n\n"
            f"```\n{session.to_dsl()}\n```\n\n"
            f"Summary: {len(session._joints)} joints, {len(session._members)} members."
        )},
    ]

    return {"messages": messages, "category": 6}


def _extend_bays_example(params: StructureParams) -> Optional[dict]:
    """Build a frame, then extend it with additional bays."""
    if params.n_bays_x < 2:
        params.n_bays_x = 2

    session = ToolSession()
    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]

    create_column_grid(session,
        grid_x=grid_x, grid_z=[0.0], base_y=0.0,
        top_y=params.height_per_level, section=params.column_section)

    top_joints = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x)
    if len(top_joints) >= 2:
        pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
        create_beam_spans(session, joint_pairs=pairs, section=params.beam_section_x)

    base_joints = sorted(session.find_joints_at_elevation(0.0))
    add_supports(session, joints=base_joints, support_type=params.support_type)

    # Extension: add 1-2 more bays
    n_extra = random.choice([1, 2])
    max_x = params.n_bays_x * params.span_x
    ext_grid_x = [max_x + (i + 1) * params.span_x for i in range(n_extra)]
    ext_grid_x = [max_x] + ext_grid_x  # Include the existing edge

    col_args = {
        "grid_x": ext_grid_x,
        "grid_z": [0.0],
        "base_y": 0.0,
        "top_y": params.height_per_level,
        "section": params.column_section,
    }
    col_result = create_column_grid(session, **col_args)

    # Connect with beams
    all_top = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x)

    # Find the new beam spans (from old last column to new columns)
    new_pairs = []
    for i in range(len(all_top) - 1):
        j1 = all_top[i]
        j2 = all_top[i + 1]
        # Check if this pair already has a member
        exists = False
        for mid, m in session._members.items():
            if (m.start_joint == j1 and m.end_joint == j2) or \
               (m.start_joint == j2 and m.end_joint == j1):
                exists = True
                break
        if not exists:
            new_pairs.append((j1, j2))

    if not new_pairs:
        return None

    beam_args = {"joint_pairs": new_pairs, "section": params.beam_section_x}
    beam_result = create_beam_spans(session, **beam_args)

    # Supports for new base joints
    new_base = []
    for jid in session.find_joints_at_elevation(0.0):
        j = session.get_joint(jid)
        if j.x > max_x - 0.01:
            new_base.append(jid)

    if new_base:
        sup_args = {"joints": sorted(new_base), "support_type": params.support_type}
        sup_result = add_supports(session, **sup_args)

    script = session.to_script()
    val_result = validate(script)
    if not val_result.is_valid:
        return None

    new_total_x = max_x + n_extra * params.span_x
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Extend the existing {params.n_bays_x}-bay frame by {n_extra} more "
            f"bay{'s' if n_extra > 1 else ''} ({params.span_x}m each) on the positive X side."
        )},
        {"role": "assistant", "content": (
            f"<think>\n"
            f"Extending from {max_x}m to {new_total_x}m.\n"
            f"Need {n_extra} new column(s) + beams connecting to existing frame.\n"
            f"The existing edge at X={max_x} already has joints.\n"
            f"</think>"
        ), "tool_calls": [_fmt_call("call_1", "create_column_grid", col_args)]},
        {"role": "tool", "tool_call_id": "call_1", "content": col_result},
        {"role": "assistant", "content": "New columns added. Connecting with beams.",
         "tool_calls": [_fmt_call("call_2", "create_beam_spans", beam_args)]},
        {"role": "tool", "tool_call_id": "call_2", "content": beam_result},
        {"role": "assistant", "content": (
            f"Frame extended by {n_extra} bay(s). Total length: {new_total_x}m.\n\n"
            f"```\n{session.to_dsl()}\n```\n\n"
            f"Summary: {len(session._joints)} joints, {len(session._members)} members."
        )},
    ]

    return {"messages": messages, "category": 6}


# ── Helpers ──

_SYSTEM_PROMPT = (
    "You are a structural engineering assistant. You design structures by "
    "reasoning step-by-step about the requirements and calling tools to compute "
    "exact geometry. You never write coordinates or joint numbers directly — "
    "you delegate all spatial computation to Python tools. "
    "After calling all necessary tools, call assemble_script to produce "
    "a validated STAAD Pro script."
)


def _fmt_call(call_id: str, tool_name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(args)},
    }
