"""
Tool registry — maps tool names to implementations and exports OpenAI function schemas.
"""

from __future__ import annotations
from .session import ToolSession
from .column_grid import create_column_grid
from .beam_grid import create_beam_grid, create_beam_spans
from .bracing import add_bracing
from .loads import add_supports, add_member_load, place_equipment
from .assembler import assemble_script


# ── Tool function map ──

TOOL_FUNCTIONS = {
    "create_column_grid": create_column_grid,
    "create_beam_grid": create_beam_grid,
    "create_beam_spans": create_beam_spans,
    "add_bracing": add_bracing,
    "add_supports": add_supports,
    "add_member_load": add_member_load,
    "place_equipment": place_equipment,
    "assemble_script": assemble_script,
}


# ── OpenAI-compatible function schemas ──

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "create_column_grid",
            "description": "Create vertical columns at every intersection of grid_x and grid_z lines. Returns joint IDs and member IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_x": {"type": "array", "items": {"type": "number"}, "description": "X-coordinates of column lines in meters"},
                    "grid_z": {"type": "array", "items": {"type": "number"}, "description": "Z-coordinates of column lines in meters"},
                    "base_y": {"type": "number", "description": "Base elevation in meters (usually 0)"},
                    "top_y": {"type": "number", "description": "Top elevation in meters"},
                    "section": {"type": "string", "description": "AISC section, e.g. W12X65"},
                },
                "required": ["grid_x", "grid_z", "base_y", "top_y", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_beam_grid",
            "description": "Create a regular grid of beams at a given elevation. Main beams at grid lines, optional secondary beams at finer spacing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "elevation": {"type": "number", "description": "Y-elevation for the beam grid"},
                    "x_positions": {"type": "array", "items": {"type": "number"}, "description": "X-coordinates of main grid lines"},
                    "z_positions": {"type": "array", "items": {"type": "number"}, "description": "Z-coordinates of main grid lines"},
                    "main_section": {"type": "string", "description": "Section for main beams, e.g. W16X36"},
                    "secondary_x_positions": {"type": "array", "items": {"type": "number"}, "description": "Optional secondary X grid lines"},
                    "secondary_z_positions": {"type": "array", "items": {"type": "number"}, "description": "Optional secondary Z grid lines"},
                    "secondary_section": {"type": "string", "description": "Section for secondary beams"},
                },
                "required": ["elevation", "x_positions", "z_positions", "main_section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_beam_spans",
            "description": "Create individual beam spans between specified joint pairs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint_pairs": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "description": "List of [start_joint, end_joint] pairs"},
                    "section": {"type": "string", "description": "Beam section designation"},
                },
                "required": ["joint_pairs", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_bracing",
            "description": "Add bracing in vertical bays. Each bay is defined by 4 corner joint IDs: [bottom_left, bottom_right, top_left, top_right].",
            "parameters": {
                "type": "object",
                "properties": {
                    "bay_corners": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}, "description": "List of bays, each [BL, BR, TL, TR joint IDs]"},
                    "brace_type": {"type": "string", "enum": ["X", "K", "V", "CHEVRON"], "description": "Bracing type"},
                    "section": {"type": "string", "description": "Bracing section, e.g. L4X4X3/8"},
                },
                "required": ["bay_corners"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_supports",
            "description": "Add boundary condition supports at specified joints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "joints": {"type": "array", "items": {"type": "integer"}, "description": "Joint IDs to support"},
                    "support_type": {"type": "string", "enum": ["FIXED", "PINNED"], "description": "Support type"},
                },
                "required": ["joints"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_member_load",
            "description": "Add distributed or concentrated loads to members.",
            "parameters": {
                "type": "object",
                "properties": {
                    "members": {"type": "array", "items": {"type": "integer"}, "description": "Member IDs to load"},
                    "load_type": {"type": "string", "enum": ["UNI", "CON", "LIN", "TRAP"], "description": "Load type"},
                    "direction": {"type": "string", "enum": ["GX", "GY", "GZ"], "description": "Load direction (GY = vertical downward)"},
                    "value": {"type": "number", "description": "Load magnitude in kN/m (UNI) or kN (CON). Negative = downward for GY."},
                    "case_id": {"type": "integer", "description": "Load case ID (default 1)"},
                },
                "required": ["members", "load_type", "direction", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_equipment",
            "description": "Place an equipment point load at a specified location on the platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kn": {"type": "number", "description": "Equipment weight in kN"},
                    "x": {"type": "number", "description": "X-coordinate of equipment center"},
                    "z": {"type": "number", "description": "Z-coordinate of equipment center"},
                    "elevation": {"type": "number", "description": "Platform elevation (Y)"},
                    "case_id": {"type": "integer", "description": "Load case ID (default 1)"},
                },
                "required": ["weight_kn", "x", "z", "elevation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assemble_script",
            "description": "Assemble all accumulated elements into a validated STAAD Pro script. Call this as the final step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Structure title"},
                },
            },
        },
    },
]


def execute_tool(session: ToolSession, tool_name: str, arguments: dict) -> str:
    """
    Execute a tool by name with the given arguments.

    Args:
        session: The active tool session.
        tool_name: Name of the tool to execute.
        arguments: Keyword arguments for the tool.

    Returns:
        Human-readable result string.
    """
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return f"ERROR: Unknown tool '{tool_name}'. Available tools: {list(TOOL_FUNCTIONS.keys())}"

    try:
        return func(session, **arguments)
    except Exception as e:
        return f"ERROR executing {tool_name}: {e}"
