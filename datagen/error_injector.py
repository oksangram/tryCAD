"""
Error injection generator.

Creates training examples where an intentional error is introduced
into a design, and the model must:
1. Detect the error via validation
2. Diagnose the root cause
3. Fix it with a corrective tool call

This teaches the model to self-correct.
"""

from __future__ import annotations
import random
import copy
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
from .reasoning_generator import generate_reasoning


# ── Error types ──

ERROR_TYPES = [
    "missing_supports",
    "missing_loads",
    "wrong_section",
    "missing_beams",
]


def generate_error_example(params: Optional[StructureParams] = None) -> Optional[dict]:
    """
    Generate a training example with an intentional error and correction.

    Returns a dict with messages showing:
    1. Normal design process (with error)
    2. Validation failure
    3. Diagnosis + corrective tool call
    4. Successful validation
    """
    if params is None:
        params = sample_random_structure()

    error_type = random.choice(ERROR_TYPES)

    if error_type == "missing_supports":
        return _missing_supports_example(params)
    elif error_type == "missing_loads":
        return _missing_loads_example(params)
    elif error_type == "wrong_section":
        return _wrong_section_example(params)
    elif error_type == "missing_beams":
        return _missing_beams_example(params)

    return None


def _missing_supports_example(params: StructureParams) -> Optional[dict]:
    """
    Build a structure but forget to add supports.
    The model should detect 'No supports defined' and fix it.
    """
    session = ToolSession()
    calls = []

    # Use portal frame for simplicity
    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]

    # Step 1: Columns
    args = {
        "grid_x": grid_x,
        "grid_z": [0.0],
        "base_y": 0.0,
        "top_y": params.height_per_level,
        "section": params.column_section,
    }
    result = create_column_grid(session, **args)
    calls.append({"tool": "create_column_grid", "args": args, "result": result})

    # Step 2: Beams
    top_joints = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x
    )
    if len(top_joints) >= 2:
        pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
        args = {"joint_pairs": pairs, "section": params.beam_section_x}
        result = create_beam_spans(session, **args)
        calls.append({"tool": "create_beam_spans", "args": args, "result": result})

    # Step 3: Loads (but NO supports — this is the error)
    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.tributary_width
    beam_members = _find_horizontal_members(session, params.height_per_level)
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

    # Validation should fail
    script = session.to_script()
    val_result = validate(script)

    # Now fix it
    base_joints = sorted(
        session.find_joints_at_elevation(0.0),
        key=lambda jid: session.get_joint(jid).x
    )
    fix_args = {"joints": base_joints, "support_type": params.support_type}
    fix_result = add_supports(session, **fix_args)

    # Re-validate
    script2 = session.to_script()
    val_result2 = validate(script2)
    if not val_result2.is_valid:
        return None

    nl_spec = generate_nl_spec(params, detail="full")
    reasoning = generate_reasoning(params)

    # Build messages showing the error → diagnosis → fix flow
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": nl_spec},
    ]

    # Initial tool calls (with error)
    for i, tc in enumerate(calls):
        call_id = f"call_{i + 1}"
        content = reasoning if i == 0 else f"Good. Now calling {tc['tool']}."
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [_fmt_call(call_id, tc["tool"], tc["args"])],
        })
        messages.append({"role": "tool", "tool_call_id": call_id, "content": tc["result"]})

    # Diagnosis step
    fix_call_id = f"call_{len(calls) + 1}"
    diagnosis = (
        "I notice I haven't added supports yet. The structure needs "
        f"{params.support_type} supports at all base joints (elevation 0.0m). "
        "Let me add them now."
    )
    messages.append({
        "role": "assistant",
        "content": diagnosis,
        "tool_calls": [_fmt_call(fix_call_id, "add_supports", fix_args)],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": fix_call_id,
        "content": fix_result,
    })

    # Final
    messages.append({
        "role": "assistant",
        "content": (
            f"Structure complete with supports added. "
            f"Validated STAAD Pro script:\n\n"
            f"```\n{session.to_dsl()}\n```\n\n"
            f"Summary: {len(session._joints)} joints, {len(session._members)} members."
        ),
    })

    return {"messages": messages, "category": 4}


def _missing_loads_example(params: StructureParams) -> Optional[dict]:
    """
    Build a structure but forget loads.
    The model should add the missing load case.
    """
    session = ToolSession()
    calls = []

    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]

    args = {
        "grid_x": grid_x,
        "grid_z": [0.0],
        "base_y": 0.0,
        "top_y": params.height_per_level,
        "section": params.column_section,
    }
    result = create_column_grid(session, **args)
    calls.append({"tool": "create_column_grid", "args": args, "result": result})

    top_joints = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x
    )
    if len(top_joints) >= 2:
        pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
        args = {"joint_pairs": pairs, "section": params.beam_section_x}
        result = create_beam_spans(session, **args)
        calls.append({"tool": "create_beam_spans", "args": args, "result": result})

    # Add supports but NO loads
    base_joints = sorted(session.find_joints_at_elevation(0.0))
    args = {"joints": base_joints, "support_type": params.support_type}
    result = add_supports(session, **args)
    calls.append({"tool": "add_supports", "args": args, "result": result})

    # Now add the missing load
    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.tributary_width
    beam_members = _find_horizontal_members(session, params.height_per_level)
    if not beam_members:
        return None

    fix_args = {
        "members": sorted(beam_members),
        "load_type": "UNI",
        "direction": "GY",
        "value": round(udl, 2),
        "case_id": 1,
    }
    fix_result = add_member_load(session, **fix_args)

    script = session.to_script()
    val_result = validate(script)
    if not val_result.is_valid:
        return None

    nl_spec = generate_nl_spec(params, detail="full")
    reasoning = generate_reasoning(params)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": nl_spec},
    ]

    for i, tc in enumerate(calls):
        call_id = f"call_{i + 1}"
        content = reasoning if i == 0 else f"Good. Now calling {tc['tool']}."
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [_fmt_call(call_id, tc["tool"], tc["args"])],
        })
        messages.append({"role": "tool", "tool_call_id": call_id, "content": tc["result"]})

    fix_call_id = f"call_{len(calls) + 1}"
    messages.append({
        "role": "assistant",
        "content": (
            "I still need to apply the design loads. The specification calls for "
            f"DL={params.load_dl_kpa} kPa + LL={params.load_ll_kpa} kPa "
            f"with tributary width {params.tributary_width}m, giving a UDL of "
            f"{round(udl, 2)} kN/m on all beams."
        ),
        "tool_calls": [_fmt_call(fix_call_id, "add_member_load", fix_args)],
    })
    messages.append({"role": "tool", "tool_call_id": fix_call_id, "content": fix_result})

    messages.append({
        "role": "assistant",
        "content": (
            f"Structure complete.\n\n```\n{session.to_dsl()}\n```\n\n"
            f"Summary: {len(session._joints)} joints, {len(session._members)} members."
        ),
    })

    return {"messages": messages, "category": 4}


def _wrong_section_example(params: StructureParams) -> Optional[dict]:
    """
    Build correctly but note that a section choice is suboptimal.
    Shows the model reasoning about engineering adequacy.
    """
    # Use a deliberately undersized section, then note the concern
    session = ToolSession()
    calls = []

    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]
    undersized = "W8X18"  # Smallest section

    args = {
        "grid_x": grid_x,
        "grid_z": [0.0],
        "base_y": 0.0,
        "top_y": params.height_per_level,
        "section": undersized,
    }
    result = create_column_grid(session, **args)
    calls.append({"tool": "create_column_grid", "args": args, "result": result})

    top_joints = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x
    )
    if len(top_joints) >= 2:
        pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
        args2 = {"joint_pairs": pairs, "section": params.beam_section_x}
        result2 = create_beam_spans(session, **args2)
        calls.append({"tool": "create_beam_spans", "args": args2, "result": result2})

    base_joints = sorted(session.find_joints_at_elevation(0.0))
    args3 = {"joints": base_joints, "support_type": params.support_type}
    result3 = add_supports(session, **args3)
    calls.append({"tool": "add_supports", "args": args3, "result": result3})

    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.tributary_width
    beam_members = _find_horizontal_members(session, params.height_per_level)
    if beam_members:
        args4 = {
            "members": sorted(beam_members),
            "load_type": "UNI",
            "direction": "GY",
            "value": round(udl, 2),
            "case_id": 1,
        }
        result4 = add_member_load(session, **args4)
        calls.append({"tool": "add_member_load", "args": args4, "result": result4})

    script = session.to_script()
    val_result = validate(script)
    if not val_result.is_valid:
        return None

    nl_spec = generate_nl_spec(params, detail="full")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": nl_spec},
    ]

    reasoning_with_concern = (
        "<think>\n"
        f"The specification calls for {params.column_section} columns, but I initially "
        f"used {undersized}. For a {params.height_per_level}m column height with "
        f"tributary area {params.span_x * params.tributary_width}m², {undersized} "
        f"is likely undersized. I should use {params.column_section} as specified. "
        f"However, since the user wants to see the analysis first, I'll note this "
        f"concern and proceed with the specified section.\n"
        "</think>"
    )

    for i, tc in enumerate(calls):
        call_id = f"call_{i + 1}"
        content = reasoning_with_concern if i == 0 else f"Good. Now calling {tc['tool']}."
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [_fmt_call(call_id, tc["tool"], tc["args"])],
        })
        messages.append({"role": "tool", "tool_call_id": call_id, "content": tc["result"]})

    messages.append({
        "role": "assistant",
        "content": (
            f"**Note:** I used {undersized} for columns, but the specification requests "
            f"{params.column_section}. For {params.height_per_level}m columns under "
            f"the given loading, {params.column_section} would be more appropriate. "
            f"Consider re-running with the correct section.\n\n"
            f"```\n{session.to_dsl()}\n```\n\n"
            f"Summary: {len(session._joints)} joints, {len(session._members)} members."
        ),
    })

    return {"messages": messages, "category": 5}


def _missing_beams_example(params: StructureParams) -> Optional[dict]:
    """
    Build columns and supports but forget beams.
    The model catches the issue and adds beam spans.
    """
    session = ToolSession()
    calls = []

    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]

    args = {
        "grid_x": grid_x,
        "grid_z": [0.0],
        "base_y": 0.0,
        "top_y": params.height_per_level,
        "section": params.column_section,
    }
    result = create_column_grid(session, **args)
    calls.append({"tool": "create_column_grid", "args": args, "result": result})

    # Skip beams (error) — go straight to supports
    base_joints = sorted(session.find_joints_at_elevation(0.0))
    args2 = {"joints": base_joints, "support_type": params.support_type}
    result2 = add_supports(session, **args2)
    calls.append({"tool": "add_supports", "args": args2, "result": result2})

    # Now fix: add the beams
    top_joints = sorted(
        session.find_joints_at_elevation(params.height_per_level),
        key=lambda jid: session.get_joint(jid).x
    )
    if len(top_joints) < 2:
        return None

    pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
    fix_args = {"joint_pairs": pairs, "section": params.beam_section_x}
    fix_result = create_beam_spans(session, **fix_args)

    # Add loads
    udl = -1 * (params.load_dl_kpa + params.load_ll_kpa) * params.tributary_width
    beam_members = _find_horizontal_members(session, params.height_per_level)
    if not beam_members:
        return None

    load_args = {
        "members": sorted(beam_members),
        "load_type": "UNI",
        "direction": "GY",
        "value": round(udl, 2),
        "case_id": 1,
    }
    load_result = add_member_load(session, **load_args)

    script = session.to_script()
    val_result = validate(script)
    if not val_result.is_valid:
        return None

    nl_spec = generate_nl_spec(params, detail="full")
    reasoning = generate_reasoning(params)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": nl_spec},
    ]

    for i, tc in enumerate(calls):
        call_id = f"call_{i + 1}"
        content = reasoning if i == 0 else f"Good. Now calling {tc['tool']}."
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [_fmt_call(call_id, tc["tool"], tc["args"])],
        })
        messages.append({"role": "tool", "tool_call_id": call_id, "content": tc["result"]})

    # Correction
    fix_id = f"call_{len(calls) + 1}"
    load_id = f"call_{len(calls) + 2}"

    messages.append({
        "role": "assistant",
        "content": (
            "Wait — I skipped the beam spans. The columns are up but there are no "
            "horizontal members connecting the tops. I need to add beams before "
            "applying loads."
        ),
        "tool_calls": [_fmt_call(fix_id, "create_beam_spans", fix_args)],
    })
    messages.append({"role": "tool", "tool_call_id": fix_id, "content": fix_result})

    messages.append({
        "role": "assistant",
        "content": "Beams added. Now applying the design loads.",
        "tool_calls": [_fmt_call(load_id, "add_member_load", load_args)],
    })
    messages.append({"role": "tool", "tool_call_id": load_id, "content": load_result})

    messages.append({
        "role": "assistant",
        "content": (
            f"Structure complete.\n\n```\n{session.to_dsl()}\n```\n\n"
            f"Summary: {len(session._joints)} joints, {len(session._members)} members."
        ),
    })

    return {"messages": messages, "category": 5}


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


def _find_horizontal_members(session: ToolSession, elevation: float) -> list[int]:
    members = []
    for mid, m in session._members.items():
        j1 = session.get_joint(m.start_joint)
        j2 = session.get_joint(m.end_joint)
        if j1 and j2:
            if (abs(j1.y - elevation) < 0.01 and abs(j2.y - elevation) < 0.01
                    and abs(j1.y - j2.y) < 0.01):
                members.append(mid)
    return members
