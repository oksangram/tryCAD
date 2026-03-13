"""
End-to-End Extraction Pipeline: Drawing Image → STAAD.Pro Script

Pipeline stages:
  1. Qwen3-VL: Image → JSON geometry (mm)
  2. Qwen2.5-72B: JSON → Draft STAAD script
  3. Parser + Validator: QC check
  4. If errors → LLM correction loop (max 2 retries)
  5. Output: validated .std file

Usage:
    python -m extraction.pipeline --image drawing.png --output output.std

RunPod usage:
    python -m extraction.pipeline \
        --image data/extraction/level3/images/level3_00100.png \
        --vl-lora /workspace/models/extraction_vl_lora/checkpoint-300 \
        --llm-lora /workspace/models/ift_lora/checkpoint-100
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Suppress transformers deprecation warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Stage 1: Vision Extraction ──────────────────────────────────

def extract_geometry(image_path: str, base_model: str, lora_path: str,
                     max_tokens: int = 4096) -> dict:
    """Run Qwen3-VL on the image to extract JSON geometry."""
    import torch
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    print(f"\n[Stage 1] Extracting geometry from: {image_path}")

    model, tokenizer = FastVisionModel.from_pretrained(
        base_model, load_in_4bit=True,
    )

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  LoRA loaded: {lora_path}")

    FastVisionModel.for_inference(model)

    messages = [
        {"role": "system", "content": (
            "You are a structural engineering drawing analyzer. "
            "Extract all geometric data and output a single JSON object."
        )},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text", "text": (
                "Parse both views of this composite structure. "
                "Output the complete JSON specification with coordinates in mm."
            )},
        ]},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = tokenizer(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.1, do_sample=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )[0].strip()

    # Parse the JSON
    geometry = json.loads(output_text)
    n_cols = len(geometry.get("columns", []))
    n_beams = len(geometry.get("beams", []))
    print(f"  ✅ Extracted: {n_cols} columns, {n_beams} beams")
    return geometry


# ── Stage 2: LLM STAAD Generation ──────────────────────────────

def generate_staad(geometry_json: dict, base_model: str,
                   lora_path: str, max_tokens: int = 4096) -> str:
    """Use Qwen2.5 to convert JSON geometry to STAAD script."""
    import torch
    from unsloth import FastLanguageModel

    print(f"\n[Stage 2] Generating STAAD script...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model, max_seq_length=8192, load_in_4bit=True,
    )

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  LoRA loaded: {lora_path}")

    FastLanguageModel.for_inference(model)

    prompt = (
        "Here is the geometry extracted from a structural drawing:\n"
        f"```json\n{json.dumps(geometry_json, indent=2)}\n```\n\n"
        "Convert this geometry into a valid STAAD.Pro input file. Rules:\n"
        "- Convert all mm dimensions to meters\n"
        "- Use STAAD SPACE format with UNIT METER KN\n"
        "- Include JOINT COORDINATES, MEMBER INCIDENCES, MEMBER PROPERTY, SUPPORT, LOADING, PERFORM ANALYSIS, FINISH\n"
        "- Add FIXED supports at base joints (Y=0)\n"
        "- Add a uniform dead+live load of -10 kN/m on all beams\n"
        "- Output ONLY the STAAD script inside a code block, no explanation"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True,
                           skip_special_tokens=True)

    print("  Generating (streaming):")
    print("  " + "─" * 50)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.1, do_sample=False,
            streamer=streamer,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )[0].strip()

    # Extract code block if present
    if "```" in output_text:
        parts = output_text.split("```")
        for part in parts[1:]:
            # Skip language identifier line
            lines = part.strip().split("\n")
            if lines[0].lower() in ("", "plaintext", "staad", "text"):
                lines = lines[1:]
            code = "\n".join(lines)
            if "STAAD" in code.upper():
                return code.strip()

    return output_text


# ── Stage 3: Validation ─────────────────────────────────────────

def validate_script(staad_text: str) -> tuple[bool, str]:
    """Parse and validate the STAAD script. Returns (is_valid, report)."""
    from dsl.parser import parse, ParseError
    from validation.validator import validate

    print(f"\n[Stage 3] Validating STAAD script...")

    try:
        script = parse(staad_text)
    except (ParseError, Exception) as e:
        report = f"PARSE ERROR: {e}"
        print(f"  ❌ {report}")
        return False, report

    result = validate(script)

    report_lines = []
    if result.errors:
        report_lines.append("ERRORS:")
        for err in result.errors:
            report_lines.append(f"  - {err}")
    if result.warnings:
        report_lines.append("WARNINGS:")
        for warn in result.warnings:
            report_lines.append(f"  - {warn}")

    report = "\n".join(report_lines) if report_lines else "All checks passed"

    if result.is_valid:
        print(f"  ✅ Valid! {len(script.joints)} joints, {len(script.members)} members")
        if result.warnings:
            for w in result.warnings:
                print(f"  ⚠️  {w}")
    else:
        print(f"  ❌ Invalid:")
        for e in result.errors:
            print(f"     {e}")

    return result.is_valid, report


# ── Stage 4: Correction Loop ────────────────────────────────────

def correct_script(staad_text: str, error_report: str,
                   geometry_json: dict, base_model: str,
                   lora_path: str) -> str:
    """Ask the LLM to fix the errors in the script."""
    import torch
    from unsloth import FastLanguageModel

    print(f"\n[Stage 4] Asking LLM to fix errors...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model, max_seq_length=8192, load_in_4bit=True,
    )
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)

    prompt = (
        f"This STAAD.Pro script has validation errors:\n\n"
        f"```\n{staad_text}\n```\n\n"
        f"Errors found:\n{error_report}\n\n"
        f"Original geometry (mm):\n```json\n{json.dumps(geometry_json, indent=2)}\n```\n\n"
        f"Fix the errors and output ONLY the corrected STAAD script in a code block."
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=4096,
            temperature=0.1, do_sample=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )[0].strip()

    # Extract code block
    if "```" in output_text:
        parts = output_text.split("```")
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if lines[0].lower() in ("", "plaintext", "staad", "text"):
                lines = lines[1:]
            code = "\n".join(lines)
            if "STAAD" in code.upper() or "JOINT" in code.upper():
                return code.strip()

    return output_text


# ── Main Pipeline ────────────────────────────────────────────────

def run_pipeline(image_path: str,
                 vl_base: str = "unsloth/Qwen3-VL-8B-Instruct",
                 vl_lora: str = None,
                 llm_base: str = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
                 llm_lora: str = None,
                 output_path: str = None,
                 max_retries: int = 2) -> str:
    """Run the full extraction pipeline."""

    # Set HF cache
    os.environ["HF_HOME"] = "/workspace/huggingface_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"
    os.environ["HF_HUB_CACHE"] = "/workspace/huggingface_cache"

    print("=" * 60)
    print("STRUCTURAL DRAWING → STAAD.Pro PIPELINE")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"VL model: {vl_base} + {vl_lora or 'no LoRA'}")
    print(f"LLM model: {llm_base} + {llm_lora or 'no LoRA'}")

    # Stage 1: Extract geometry
    geometry = extract_geometry(image_path, vl_base, vl_lora)

    # Free VL model memory
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    # Stage 2: Generate STAAD
    staad_text = generate_staad(geometry, llm_base, llm_lora)

    # Free LLM memory
    torch.cuda.empty_cache()
    gc.collect()

    # Stage 3: Validate
    is_valid, report = validate_script(staad_text)

    # Stage 4: Correction loop
    retries = 0
    while not is_valid and retries < max_retries:
        retries += 1
        print(f"\n{'─' * 60}")
        print(f"Correction attempt {retries}/{max_retries}")
        staad_text = correct_script(staad_text, report, geometry,
                                     llm_base, llm_lora)
        torch.cuda.empty_cache()
        gc.collect()
        is_valid, report = validate_script(staad_text)

    # Output
    if output_path is None:
        output_path = Path(image_path).stem + ".std"

    with open(output_path, "w") as f:
        f.write(staad_text)

    print(f"\n{'=' * 60}")
    if is_valid:
        print(f"✅ PIPELINE COMPLETE — Valid STAAD script saved to: {output_path}")
    else:
        print(f"⚠️  PIPELINE COMPLETE — Script saved (with warnings) to: {output_path}")
        print(f"   Remaining issues: {report}")
    print(f"{'=' * 60}")

    return staad_text


def main():
    parser = argparse.ArgumentParser(
        description="Extract structural geometry from drawing → STAAD.Pro")
    parser.add_argument("--image", required=True,
                        help="Path to the structural drawing image")
    parser.add_argument("--output", default=None,
                        help="Output .std file path")
    parser.add_argument("--vl-base", default="unsloth/Qwen3-VL-8B-Instruct",
                        help="Vision-Language base model")
    parser.add_argument("--vl-lora", default=None,
                        help="Path to VL LoRA adapter")
    parser.add_argument("--llm-base",
                        default="unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
                        help="LLM base model")
    parser.add_argument("--llm-lora", default=None,
                        help="Path to LLM LoRA adapter")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max correction retries on validation failure")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        vl_base=args.vl_base,
        vl_lora=args.vl_lora,
        llm_base=args.llm_base,
        llm_lora=args.llm_lora,
        output_path=args.output,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
