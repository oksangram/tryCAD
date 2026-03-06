"""
Prepare IFT (Instruction Fine-Tuning) data.

Converts the JSONL training data into the Qwen2.5 chat template format
with proper tool call formatting for Unsloth training.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.registry import TOOLS_SCHEMA


def prepare_ift_data(
    ift_input: str = "data/ift_train.jsonl",
    output_train: str = "data/ift_train_prepared.jsonl",
    output_eval: str = "data/ift_eval_prepared.jsonl",
    eval_fraction: float = 0.05,
    seed: int = 42,
):
    """
    Prepare IFT data for Unsloth SFT training.

    Reads the raw JSONL from the pipeline and:
    1. Adds the tool schema to each system message
    2. Ensures proper Qwen2.5 chat format
    3. Splits into train/eval
    4. Writes as HuggingFace-compatible JSONL
    """
    import random
    random.seed(seed)

    os.makedirs(os.path.dirname(output_train), exist_ok=True)

    # Read all examples
    examples = []
    with open(ift_input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Augment system messages with tool schema
    tool_schema_str = json.dumps(TOOLS_SCHEMA, indent=2)
    augmented = []

    for ex in examples:
        messages = ex.get("messages", [])
        if not messages:
            continue

        # Augment system message with available tools
        if messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            messages[0]["content"] = (
                f"{system_content}\n\n"
                f"You have access to the following tools:\n"
                f"{tool_schema_str}\n\n"
                f"When you need to call a tool, use the tool_calls format. "
                f"Always reason before calling tools."
            )

        augmented.append({"messages": messages})

    # Shuffle and split
    random.shuffle(augmented)
    n_eval = max(1, int(len(augmented) * eval_fraction))
    eval_data = augmented[:n_eval]
    train_data = augmented[n_eval:]

    # Write train
    with open(output_train, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write eval
    with open(output_eval, "w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"IFT data prepared:")
    print(f"  Train: {len(train_data)} examples → {output_train}")
    print(f"  Eval:  {len(eval_data)} examples → {output_eval}")

    return {
        "train_count": len(train_data),
        "eval_count": len(eval_data),
        "train_path": output_train,
        "eval_path": output_eval,
    }


if __name__ == "__main__":
    prepare_ift_data()
