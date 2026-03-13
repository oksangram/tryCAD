"""
Training data formatter for Qwen3-VL fine-tuning.

Converts manifest JSONL files (from synthetic_drawings.py) into the
Qwen3-VL multimodal chat format required by Unsloth/HuggingFace SFT.

Output format (per example):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "image", "image": "file:///absolute/path/to/image.png"},
      {"type": "text", "text": "Extract ..."}
    ]},
    {"role": "assistant", "content": "<json output>"}
  ]
}

Usage:
    python -m extraction_training.prepare_extraction_data \
        --input data/extraction \
        --output data/extraction_vl_train.jsonl
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


# ── System prompts ──

SYSTEM_PROMPT = (
    "You are a structural engineering drawing analyzer. "
    "Given an image of a structural drawing, extract all geometric data, "
    "grid coordinates, member locations, and section assignments. "
    "Output a single JSON object with the complete structural specification."
)

# ── User prompt templates (varied for robustness) ──

USER_PROMPTS_L1 = [
    "Extract the member details from this structural drawing.",
    "What structural member is shown in this drawing? Give section, quantity, and dimensions as JSON.",
    "Read this drawing and output the member specification as structured JSON.",
    "Identify the structural element in this image. Output coordinates and section.",
    "Parse this engineering drawing. Extract member role, section, quantity, and length.",
]

USER_PROMPTS_L2 = [
    "Extract the structural grid, member locations, and sections from this drawing.",
    "Analyze this structural frame drawing. Output the grid coordinates, column positions, beam spans, and all section assignments as JSON.",
    "Read this elevation/plan drawing. Extract the full structural specification including grid lines, member coordinates, and sections.",
    "Parse this engineering drawing and output the complete structural geometry as JSON, including grid positions in mm.",
    "What structure is shown? Extract bay count, spacing, height, grid coordinates, and all member sections.",
]

USER_PROMPTS_L3 = [
    "This drawing shows multiple views of a structure. Extract the complete structural specification from all views.",
    "Analyze the elevation and plan views. Output the full geometry including grid coordinates, column positions, beam locations, bracing, and stringers.",
    "Extract all structural details from these multi-view drawings. Include grid lines, member coordinates, sections, and bracing type.",
    "Parse both views of this composite structure. Output the complete JSON specification with coordinates in mm.",
    "Read this engineering drawing showing elevation and plan. Extract grid, columns, beams, stringers, and bracing.",
]

PROMPT_MAP = {1: USER_PROMPTS_L1, 2: USER_PROMPTS_L2, 3: USER_PROMPTS_L3}


def format_example(entry: dict, images_base: Path) -> dict:
    """Convert a single manifest entry to Qwen3-VL chat format."""
    level = entry["level"]
    image_rel = entry["image"]
    ground_truth = entry["ground_truth"]

    # Absolute image path for training
    image_abs = (images_base / image_rel).resolve()

    # Pick a random user prompt for this level
    user_text = random.choice(PROMPT_MAP.get(level, USER_PROMPTS_L2))

    # Format ground truth as pretty JSON for the assistant response
    assistant_json = json.dumps(ground_truth, indent=2)

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_abs}"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_json,
            },
        ]
    }


def process_manifest(manifest_path: Path, images_base: Path) -> list[dict]:
    """Process a single manifest file into training examples."""
    examples = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            example = format_example(entry, images_base)
            examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Format extraction training data for Qwen3-VL")
    parser.add_argument("--input", type=str, required=True,
                        help="Base directory containing level1/level2/level3 subdirs")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction to hold out for evaluation (default 0.1)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    all_examples = []

    # Process each level's manifest
    for level in [1, 2, 3]:
        # Handle nested directory structure from generator
        manifest_candidates = [
            input_dir / f"level{level}" / f"level{level}_manifest.jsonl",
            input_dir / f"level{level}" / f"level{level}" / f"level{level}_manifest.jsonl",
        ]

        manifest_path = None
        for candidate in manifest_candidates:
            if candidate.exists():
                manifest_path = candidate
                break

        if manifest_path is None:
            print(f"  Level {level}: no manifest found, skipping")
            continue

        images_base = manifest_path.parent
        examples = process_manifest(manifest_path, images_base)
        print(f"  Level {level}: {len(examples)} examples from {manifest_path}")
        all_examples.extend(examples)

    if not all_examples:
        print("ERROR: No examples found. Check --input directory structure.")
        return

    # Shuffle
    random.shuffle(all_examples)

    # Split train/eval
    n_eval = max(1, int(len(all_examples) * args.eval_split))
    n_train = len(all_examples) - n_eval

    train_examples = all_examples[:n_train]
    eval_examples = all_examples[n_train:]

    # Write train split
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Write eval split
    eval_path = output_path.with_name(
        output_path.stem + "_eval" + output_path.suffix)
    with open(eval_path, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Formatted {len(all_examples)} total examples")
    print(f"   Train: {n_train} → {output_path}")
    print(f"   Eval:  {n_eval} → {eval_path}")


if __name__ == "__main__":
    main()
