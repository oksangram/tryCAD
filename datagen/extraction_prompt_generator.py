"""
Extraction Prompt Generator for LLM Fine-tuning.

This script generates Instruction Fine-Tuning (IFT) examples that teach the Qwen2.5 
Tool-Calling LLM how to parse the JSON output from the Qwen3-VL extraction model 
and map it to the structural engineering tools.

It simulates the Qwen3-VL JSON output from `StructureParams`, including millimeter 
dimensions, and pairs it with the exact tool-call execution trace.
"""

from __future__ import annotations
import json
import random
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datagen.sampler import StructureParams, sample_random_structure
from datagen.toolcall_generator import generate_structure
from datagen.pipeline import SYSTEM_PROMPT, _build_messages_with_tools

def params_to_vision_json(params: StructureParams) -> dict:
    """
    Convert StructureParams to the exact JSON format output by the Qwen3-VL model.
    Note: Vision model outputs in millimeters, while params are in meters.
    """
    # Convert spans to grid lines in mm
    grid_x_mm = [int(i * params.span_x * 1000) for i in range(params.n_bays_x + 1)]
    grid_z_mm = [int(i * params.span_z * 1000) for i in range(params.n_bays_z + 1)] if params.n_bays_z > 0 else [0]
    elevations_mm = [int(i * params.height_per_level * 1000) for i in range(params.n_levels + 1)]

    columns = []
    for x in grid_x_mm:
        for z in grid_z_mm:
            columns.append({
                "grid_x_mm": x,
                "grid_z_mm": z,
                "base_mm": 0,
                "top_mm": elevations_mm[-1],
                "section": params.column_section
            })

    beams = []
    # Beams in X direction
    for i in range(params.n_bays_x):
        for z in grid_z_mm:
            for elev in elevations_mm[1:]:
                beams.append({
                    "start_mm": {"x": grid_x_mm[i], "y": elev, "z": z},
                    "end_mm": {"x": grid_x_mm[i+1], "y": elev, "z": z},
                    "section": params.beam_section_x,
                    "role": "main_beam"
                })
    
    # Beams in Z direction
    if params.n_bays_z > 0:
        for i in range(params.n_bays_z):
            for x in grid_x_mm:
                for elev in elevations_mm[1:]:
                    beams.append({
                        "start_mm": {"x": x, "y": elev, "z": grid_z_mm[i]},
                        "end_mm": {"x": x, "y": elev, "z": grid_z_mm[i+1]},
                        "section": params.beam_section_z,
                        "role": "secondary_beam"
                    })

    bracing = []
    if params.bracing_type != "NONE" and params.bracing_bays:
        for bay_idx in params.bracing_bays:
            if bay_idx < params.n_bays_x:
                for z in [grid_z_mm[0], grid_z_mm[-1]]:
                    bracing.append({
                        "bay_index": bay_idx,
                        "bay_z_mm": z,
                        "type": params.bracing_type,
                        "section": params.bracing_section
                    })

    # Mimic the vision output structure
    vision_json = {
        "views": ["elevation", "plan"] if params.n_bays_z > 0 else ["elevation"],
        "structure_type": params.structure_type,
        "grid": {
            "x_lines_mm": grid_x_mm,
            "z_lines_mm": grid_z_mm,
            "elevations_mm": elevations_mm
        },
        "columns": columns,
        "beams": beams,
    }
    if bracing:
        vision_json["bracing"] = bracing

    return vision_json

def generate_extraction_example(params: StructureParams) -> Optional[dict]:
    """Generate a full IFT example bridging vision JSON to tool calls."""
    gen_result = generate_structure(params)
    if gen_result is None:
        return None

    # This is the "user prompt" - exactly what the Vision model outputs
    vision_json = params_to_vision_json(params)
    nl_spec = (
        "Here is the geometry extracted from the structural drawing:\n"
        "```json\n"
        f"{json.dumps(vision_json, indent=2)}\n"
        "```\n"
        "Convert this geometry into a STAAD.pro model. Note that the extracted geometry "
        "is in millimeters, but you must call your tools using meters. Add default DEAD and LIVE member loads."
    )

    reasoning = (
        "<think>\n"
        "I need to build the structure specified in the JSON. The JSON gives coordinates in millimeters, "
        "so I must convert them to meters for the tools.\n"
        f"1. Column grid: X={vision_json['grid']['x_lines_mm']}mm -> {[x/1000 for x in vision_json['grid']['x_lines_mm']]}m, "
        f"Z={vision_json['grid']['z_lines_mm']}mm -> {[z/1000 for z in vision_json['grid']['z_lines_mm']]}m.\n"
        f"   Base is 0m, top is {vision_json['grid']['elevations_mm'][-1]/1000}m. Section: {params.column_section}.\n"
        "2. Beams: I will place the main beams at the specified elevations.\n"
        "3. Bracing: I will add bracing if specified in the JSON.\n"
        "4. Supports: Add FIXED or PINNED supports at the base.\n"
        "5. Loads: Add a standard dead/live load combination as requested.\n"
        "Let's execute the tool calls carefully.\n"
        "</think>"
    )

    messages = _build_messages_with_tools(nl_spec, reasoning, gen_result)
    return {"messages": messages, "category": "extraction_bridge"}


def generate_extraction_dataset(output_path: str, count: int = 500, seed: int = 42) -> None:
    """Generate a dataset of JSON->Tool examples."""
    random.seed(seed)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    examples = []
    attempts = 0
    max_attempts = count * 3
    
    print(f"Generating {count} extraction bridge examples...")
    
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        params = sample_random_structure()
        example = generate_extraction_example(params)
        
        if example is not None:
            examples.append(example)
            if len(examples) % 100 == 0:
                print(f"  Generated {len(examples)}/{count}...")
                
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            row_str = json.dumps(ex, ensure_ascii=False).replace('\n', '\\n').replace('\r', '')
            f.write(row_str + "\n")
            
    print(f"✅ Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    dataset_path = "data/ift_extraction_bridge.jsonl"
    generate_extraction_dataset(dataset_path, count)
