"""
Spatial reasoning exercise generator.

Creates Q&A-style training examples that test spatial understanding:
- "Which joints are at the top level?"
- "What is the distance between joint A and joint B?"
- "Which members are connected to joint X?"
- "How many bays span in the X direction?"

These teach the model to reason about structural geometry.
"""

from __future__ import annotations
import random
import json
import math
from typing import Optional

from tools.session import ToolSession
from tools.column_grid import create_column_grid
from tools.beam_grid import create_beam_grid, create_beam_spans
from tools.loads import add_supports

from .sampler import sample_random_structure, StructureParams


def generate_spatial_exercise(params: Optional[StructureParams] = None) -> Optional[dict]:
    """Generate a spatial reasoning Q&A example."""
    if params is None:
        params = sample_random_structure()

    exercise_type = random.choice([
        "top_joints",
        "joint_distance",
        "member_connections",
        "bay_count",
        "elevation_query",
        "bounding_box",
    ])

    # Build a small structure first
    session = ToolSession()
    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]
    grid_z = [0.0]
    if params.n_bays_z > 0:
        grid_z = [i * params.span_z for i in range(params.n_bays_z + 1)]

    create_column_grid(session,
        grid_x=grid_x, grid_z=grid_z, base_y=0.0,
        top_y=params.height_per_level, section=params.column_section)

    top_elev = params.height_per_level
    top_joints = sorted(
        session.find_joints_at_elevation(top_elev),
        key=lambda jid: session.get_joint(jid).x)

    if len(top_joints) >= 2:
        if len(grid_z) > 1:
            create_beam_grid(session,
                elevation=top_elev, x_positions=grid_x,
                z_positions=grid_z, main_section=params.beam_section_x)
        else:
            pairs = [(top_joints[i], top_joints[i + 1]) for i in range(len(top_joints) - 1)]
            create_beam_spans(session, joint_pairs=pairs, section=params.beam_section_x)

    base_joints = sorted(session.find_joints_at_elevation(0.0))
    add_supports(session, joints=base_joints, support_type=params.support_type)

    # Generate the Q&A based on exercise type
    if exercise_type == "top_joints":
        return _top_joints_exercise(session, params)
    elif exercise_type == "joint_distance":
        return _joint_distance_exercise(session, params)
    elif exercise_type == "member_connections":
        return _member_connections_exercise(session, params)
    elif exercise_type == "bay_count":
        return _bay_count_exercise(session, params)
    elif exercise_type == "elevation_query":
        return _elevation_query_exercise(session, params)
    elif exercise_type == "bounding_box":
        return _bounding_box_exercise(session, params)

    return None


def _top_joints_exercise(session: ToolSession, params: StructureParams) -> dict:
    top_elev = params.height_per_level
    top_joints = session.find_joints_at_elevation(top_elev)

    joint_list = []
    for jid in sorted(top_joints):
        j = session.get_joint(jid)
        joint_list.append(f"Joint {jid} ({j.x}, {j.y}, {j.z})")

    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"I have a {params.n_bays_x}-bay frame, {params.height_per_level}m high. "
                f"Which joints are at the top elevation?"
            )},
            {"role": "assistant", "content": (
                f"The top elevation is at Y = {top_elev}m. "
                f"There are {len(top_joints)} joints at this elevation:\n\n"
                + "\n".join(f"- {jl}" for jl in joint_list)
            )},
        ],
        "category": 8,
    }


def _joint_distance_exercise(session: ToolSession, params: StructureParams) -> Optional[dict]:
    joint_ids = list(session._joints.keys())
    if len(joint_ids) < 2:
        return None

    j1_id, j2_id = random.sample(joint_ids, 2)
    j1 = session.get_joint(j1_id)
    j2 = session.get_joint(j2_id)

    dist = math.sqrt((j1.x - j2.x)**2 + (j1.y - j2.y)**2 + (j1.z - j2.z)**2)

    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"What is the distance between Joint {j1_id} at ({j1.x}, {j1.y}, {j1.z}) "
                f"and Joint {j2_id} at ({j2.x}, {j2.y}, {j2.z})?"
            )},
            {"role": "assistant", "content": (
                f"Using the 3D distance formula:\n\n"
                f"d = sqrt(({j2.x}-{j1.x})^2 + ({j2.y}-{j1.y})^2 + ({j2.z}-{j1.z})^2)\n"
                f"d = sqrt({(j2.x-j1.x)**2:.2f} + {(j2.y-j1.y)**2:.2f} + {(j2.z-j1.z)**2:.2f})\n"
                f"d = {dist:.3f} m"
            )},
        ],
        "category": 8,
    }


def _member_connections_exercise(session: ToolSession, params: StructureParams) -> Optional[dict]:
    joint_ids = list(session._joints.keys())
    if not joint_ids:
        return None

    target_jid = random.choice(joint_ids)
    j = session.get_joint(target_jid)

    connected_members = []
    for mid, m in session._members.items():
        if m.start_joint == target_jid or m.end_joint == target_jid:
            other = m.end_joint if m.start_joint == target_jid else m.start_joint
            oj = session.get_joint(other)
            connected_members.append(f"Member {mid}: connects to Joint {other} ({oj.x}, {oj.y}, {oj.z})")

    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Which members are connected to Joint {target_jid} "
                f"at ({j.x}, {j.y}, {j.z})?"
            )},
            {"role": "assistant", "content": (
                f"Joint {target_jid} at ({j.x}, {j.y}, {j.z}) has "
                f"{len(connected_members)} connected member(s):\n\n"
                + "\n".join(f"- {cm}" for cm in connected_members)
            )},
        ],
        "category": 8,
    }


def _bay_count_exercise(session: ToolSession, params: StructureParams) -> dict:
    grid_x = [i * params.span_x for i in range(params.n_bays_x + 1)]
    total_x = params.n_bays_x * params.span_x

    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"A frame has columns at X = {grid_x}. "
                f"How many bays does it have and what is each span?"
            )},
            {"role": "assistant", "content": (
                f"The frame has **{params.n_bays_x} bay(s)**, each spanning "
                f"**{params.span_x}m**.\n\n"
                f"- Total length: {total_x}m\n"
                f"- Column lines: {len(grid_x)}\n"
                f"- Bay spans: {', '.join(f'{params.span_x}m' for _ in range(params.n_bays_x))}"
            )},
        ],
        "category": 8,
    }


def _elevation_query_exercise(session: ToolSession, params: StructureParams) -> dict:
    elevations = set()
    for jid, j in session._joints.items():
        elevations.add(round(j.y, 3))
    elevations = sorted(elevations)

    elev_data = []
    for elev in elevations:
        joints = session.find_joints_at_elevation(elev)
        elev_data.append(f"Y = {elev}m: {len(joints)} joints")

    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "What are all the elevation levels in this structure and how many joints at each?"},
            {"role": "assistant", "content": (
                f"The structure has {len(elevations)} distinct elevation level(s):\n\n"
                + "\n".join(f"- {ed}" for ed in elev_data)
            )},
        ],
        "category": 8,
    }


def _bounding_box_exercise(session: ToolSession, params: StructureParams) -> dict:
    xs = [j.x for j in session._joints.values()]
    ys = [j.y for j in session._joints.values()]
    zs = [j.z for j in session._joints.values()]

    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "What is the bounding box of this structure?"},
            {"role": "assistant", "content": (
                f"The bounding box of the structure is:\n\n"
                f"- X: {min(xs):.1f}m to {max(xs):.1f}m (width: {max(xs)-min(xs):.1f}m)\n"
                f"- Y: {min(ys):.1f}m to {max(ys):.1f}m (height: {max(ys)-min(ys):.1f}m)\n"
                f"- Z: {min(zs):.1f}m to {max(zs):.1f}m (depth: {max(zs)-min(zs):.1f}m)\n\n"
                f"Total footprint: {max(xs)-min(xs):.1f}m x {max(zs)-min(zs):.1f}m"
            )},
        ],
        "category": 8,
    }


_SYSTEM_PROMPT = (
    "You are a structural engineering assistant. You design structures by "
    "reasoning step-by-step about the requirements and calling tools to compute "
    "exact geometry. You never write coordinates or joint numbers directly — "
    "you delegate all spatial computation to Python tools. "
    "After calling all necessary tools, call assemble_script to produce "
    "a validated STAAD Pro script."
)
