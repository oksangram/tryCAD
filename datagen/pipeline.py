"""
Training data pipeline orchestrator.

Generates all 7 categories of training data, validates every example,
and outputs JSONL files for CPT and IFT.
"""

from __future__ import annotations
import json
import random
import sys
import os
from pathlib import Path
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .sampler import sample_random_structure, StructureParams
from .toolcall_generator import generate_structure
from .nl_generator import generate_nl_spec
from .reasoning_generator import generate_reasoning
from .error_injector import generate_error_example
from .modification_generator import generate_modification_example
from .spatial_exercises import generate_spatial_exercise

# System prompt for all training examples
SYSTEM_PROMPT = (
    "You are a structural engineering assistant. You design structures by "
    "reasoning step-by-step about the requirements and calling tools to compute "
    "exact geometry. You never write coordinates or joint numbers directly — "
    "you delegate all spatial computation to Python tools. "
    "After calling all necessary tools, call assemble_script to produce "
    "a validated STAAD Pro script."
)

# Clarification questions for underspecified inputs
CLARIFICATION_TEMPLATES = [
    "I need additional information before I can design this structure:\n\n"
    "1. **Exact dimensions**: What are the specific bay spans and total length required?\n"
    "2. **Height**: What is the required clear height (or structure height)?\n"
    "3. **Loading**: What dead load and live load should I design for (in kPa)?\n"
    "4. **Sections**: Do you have preferred steel sections, or should I select based on the loading?\n"
    "5. **Supports**: Should the base connections be FIXED (moment connections) or PINNED?\n"
    "6. **Bracing**: Is lateral bracing required? If so, which bays and what type (X, K, V)?",

    "Before designing, I need to clarify several requirements:\n\n"
    "1. What are the exact bay spacings and number of bays?\n"
    "2. What is the required elevation/height?\n"
    "3. What is the design loading (dead load + live load in kPa)?\n"
    "4. Are there any equipment loads to accommodate?\n"
    "5. What support conditions are appropriate for the foundation?\n"
    "6. What design code should I follow?",

    "This specification needs more detail. Please provide:\n\n"
    "1. **Geometry**: Number of bays, span lengths, and total height\n"
    "2. **Loading**: Dead load, live load, and any point loads (in kPa and kN)\n"
    "3. **Material/Sections**: Steel grade and preferred section sizes\n"
    "4. **Boundary conditions**: Support type (fixed, pinned)\n"
    "5. **Bracing requirements**: Type and location\n"
    "6. **Code requirements**: AISC, API RP 2A, etc.",
]


# ── Category wrappers for new generators ──

def generate_category_error(params: StructureParams) -> Optional[dict]:
    """Category 4-5: Error injection + correction."""
    return generate_error_example(params)


def generate_category_mod(params: StructureParams) -> Optional[dict]:
    """Category 6: Structure modification."""
    return generate_modification_example(params)


def generate_category_spatial(params: StructureParams) -> Optional[dict]:
    """Category 8: Spatial reasoning exercises."""
    return generate_spatial_exercise(params)


def generate_category_2(params: StructureParams) -> Optional[dict]:
    """Category 2: Standard structure with reasoning trace + tool calls."""
    gen_result = generate_structure(params)
    if gen_result is None:
        return None

    nl_spec = generate_nl_spec(params, detail="full")
    reasoning = generate_reasoning(params)

    messages = _build_messages_with_tools(nl_spec, reasoning, gen_result)
    return {"messages": messages, "category": 2}


def generate_category_1(params: StructureParams) -> Optional[dict]:
    """Category 1: Module-level single tool call."""
    # Pick one tool call from a generated structure
    gen_result = generate_structure(params)
    if gen_result is None or not gen_result["tool_calls"]:
        return None

    # Use only the first tool call
    first_call = gen_result["tool_calls"][0]
    tool_name = first_call["tool"]

    # Create a focused NL spec for just this module
    nl_spec = _module_level_spec(tool_name, first_call["args"], params)
    reasoning = f"<think>\nThis requires a single {tool_name} call with the specified parameters.\n</think>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl_spec},
        {"role": "assistant", "content": reasoning,
         "tool_calls": [_format_tool_call("call_1", tool_name, first_call["args"])]},
        {"role": "tool", "tool_call_id": "call_1", "content": first_call["result"]},
        {"role": "assistant", "content": f"Done. {first_call['result'].split(chr(10))[0]}"},
    ]

    return {"messages": messages, "category": 1}


def generate_category_3(params: StructureParams) -> Optional[dict]:
    """Category 3: Novel/unusual parameters with extended reasoning."""
    # Modify params to be unusual
    params.span_x = random.choice([3.5, 4.7, 7.3, 11.5, 13.0])
    params.height_per_level = random.choice([2.5, 3.7, 5.5, 9.0, 12.0])
    params.n_bays_x = random.choice([1, 2, 5, 7])

    gen_result = generate_structure(params)
    if gen_result is None:
        return None

    nl_spec = generate_nl_spec(params, detail="summary")
    reasoning = generate_reasoning(params)

    messages = _build_messages_with_tools(nl_spec, reasoning, gen_result)
    return {"messages": messages, "category": 3}


def generate_category_7(params: StructureParams) -> Optional[dict]:
    """Category 7: Underspecified input -> clarification questions."""
    nl_spec = generate_nl_spec(params, detail="underspecified")
    clarification = random.choice(CLARIFICATION_TEMPLATES)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl_spec},
        {"role": "assistant", "content": clarification},
    ]

    return {"messages": messages, "category": 7}


def _build_messages_with_tools(nl_spec: str, reasoning: str,
                                gen_result: dict) -> list[dict]:
    """Build a complete multi-turn message sequence with tool calls."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": nl_spec},
    ]

    tool_calls = gen_result["tool_calls"]
    for i, tc in enumerate(tool_calls):
        call_id = f"call_{i + 1}"

        # First call includes the reasoning trace
        content = reasoning if i == 0 else f"Proceeding to {tc['tool']}..."

        messages.append({
            "role": "assistant",
            "content": content if i == 0 else f"Good. Now calling {tc['tool']}.",
            "tool_calls": [_format_tool_call(call_id, tc["tool"], tc["args"])],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": tc["result"],
        })

    # Final assistant message with the script
    messages.append({
        "role": "assistant",
        "content": (
            f"Structure complete. Here is the validated STAAD Pro script:\n\n"
            f"```\n{gen_result['dsl_script']}\n```\n\n"
            f"Summary: {gen_result['stats']['n_joints']} joints, "
            f"{gen_result['stats']['n_members']} members."
        ),
    })

    return messages


def _format_tool_call(call_id: str, tool_name: str, args: dict) -> dict:
    """Format a tool call in OpenAI function-calling format."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(args),
        },
    }


def _module_level_spec(tool_name: str, args: dict, params: StructureParams) -> str:
    """Create a NL spec for a single tool call."""
    if tool_name == "create_column_grid":
        n_x = len(args.get("grid_x", []))
        n_z = len(args.get("grid_z", []))
        return (
            f"Create a {n_x}x{n_z} column grid. "
            f"X positions: {args['grid_x']} m. Z positions: {args['grid_z']} m. "
            f"Base at {args['base_y']}m, top at {args['top_y']}m. "
            f"Section: {args['section']}."
        )
    elif tool_name == "create_beam_grid":
        return (
            f"Create a beam grid at elevation {args['elevation']}m. "
            f"X positions: {args.get('x_positions', [])} m. "
            f"Z positions: {args.get('z_positions', [])} m. "
            f"Main section: {args['main_section']}."
        )
    elif tool_name == "add_bracing":
        return f"Add {args.get('brace_type', 'X')}-bracing with section {args.get('section', 'L4X4X3/8')}."
    elif tool_name == "add_supports":
        return f"Add {args.get('support_type', 'FIXED')} supports at joints {args['joints']}."
    elif tool_name == "add_member_load":
        return f"Add {args['load_type']} load of {args['value']} kN/m in {args['direction']} to members {args['members']}."
    return f"Execute {tool_name} with the given parameters."


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run_pipeline(output_dir: str = "data", total_examples: int = 500,
                 seed: int = 42) -> dict:
    """
    Generate training data across all categories.

    Args:
        output_dir: Directory to write output files.
        total_examples: Total number of training examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        dict with counts per category and file paths.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Category proportions (all 7 categories)
    proportions = {
        1: 0.10,   # Module-level (single tool call)
        2: 0.25,   # Standard designs with reasoning + tools
        3: 0.08,   # Novel/unusual parameters
        4: 0.12,   # Error injection + correction
        6: 0.12,   # Modification tasks
        7: 0.08,   # Underspecified -> clarification
        8: 0.10,   # Spatial reasoning exercises
    }

    results = {cat: [] for cat in proportions}
    category_targets = {cat: max(1, int(total_examples * prop)) for cat, prop in proportions.items()}

    generators = {
        1: generate_category_1,
        2: generate_category_2,
        3: generate_category_3,
        4: generate_category_error,
        6: generate_category_mod,
        7: generate_category_7,
        8: generate_category_spatial,
    }

    stats = {cat: {"generated": 0, "failed": 0} for cat in proportions}

    for cat, target in category_targets.items():
        gen_func = generators[cat]
        attempts = 0
        max_attempts = target * 3  # Allow 3x attempts for failures

        while stats[cat]["generated"] < target and attempts < max_attempts:
            attempts += 1
            params = sample_random_structure()
            try:
                example = gen_func(params)
                if example is not None:
                    results[cat].append(example)
                    stats[cat]["generated"] += 1
                else:
                    stats[cat]["failed"] += 1
            except Exception as e:
                stats[cat]["failed"] += 1


    # Write IFT data
    ift_path = os.path.join(output_dir, "ift_train.jsonl")
    all_examples = []
    for cat_examples in results.values():
        all_examples.extend(cat_examples)
    random.shuffle(all_examples)

    with open(ift_path, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Write CPT corpus (just the DSL scripts)
    cpt_path = os.path.join(output_dir, "cpt_corpus.txt")
    scripts_written = 0
    with open(cpt_path, "w", encoding="utf-8") as f:
        for cat_examples in results.values():
            for ex in cat_examples:
                # Extract DSL script from the last assistant message
                for msg in reversed(ex.get("messages", [])):
                    if msg.get("role") == "assistant" and "```" in msg.get("content", ""):
                        # Extract code block
                        content = msg["content"]
                        start = content.find("```\n") + 4
                        end = content.find("\n```", start)
                        if start > 3 and end > start:
                            f.write(content[start:end] + "\n\n")
                            scripts_written += 1
                        break

    # Summary
    total_generated = sum(s["generated"] for s in stats.values())
    print(f"\n{'='*60}")
    print(f"DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total examples: {total_generated}")
    for cat, s in sorted(stats.items()):
        print(f"  Category {cat}: {s['generated']} generated, {s['failed']} failed")
    print(f"\nIFT data: {ift_path} ({total_generated} examples)")
    print(f"CPT corpus: {cpt_path} ({scripts_written} scripts)")
    print(f"{'='*60}")

    return {
        "stats": stats,
        "total": total_generated,
        "ift_path": ift_path,
        "cpt_path": cpt_path,
    }


if __name__ == "__main__":
    # Run with a small batch for testing
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_pipeline(total_examples=n)
