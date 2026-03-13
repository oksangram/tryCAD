"""
Evaluate a Qwen3-VL model on extraction validation examples.

Tests either the base (unfinetuned) model or a fine-tuned LoRA adapter
on a few examples, printing the model's output vs the expected JSON.

Usage (base model — before fine-tuning):
    python extraction_training/eval_extraction.py \
        --eval-data data/extraction_vl_train_eval.jsonl \
        --n 5

Usage (fine-tuned model — after training):
    python extraction_training/eval_extraction.py \
        --eval-data data/extraction_vl_train_eval.jsonl \
        --lora-path /workspace/models/extraction_vl_lora \
        --n 5
"""

from __future__ import annotations
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on extraction examples")
    parser.add_argument("--eval-data", type=str, required=True,
                        help="Path to eval JSONL")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Qwen3-VL-8B-Instruct",
                        help="Base model ID")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter (omit for base model)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of examples to evaluate")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N examples (to test unseen data)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate")
    args = parser.parse_args()

    # Load examples
    examples = []
    with open(args.eval_data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    examples = examples[args.skip:args.skip + args.n]
    print(f"Evaluating on {len(examples)} examples (skipped first {args.skip})")
    print(f"Base model: {args.base_model}")
    print(f"LoRA: {args.lora_path or 'NONE (base model)'}")
    print("=" * 70)

    # Load model
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=True,
    )

    if args.lora_path:
        print(f"\nLoading LoRA adapter from {args.lora_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)

    FastVisionModel.for_inference(model)

    # Process each example
    from qwen_vl_utils import process_vision_info
    import torch

    correct = 0
    total = 0

    for i, ex in enumerate(examples):
        messages = ex["messages"]
        expected = messages[2]["content"]  # assistant response (JSON string)

        # Build input (system + user only)
        input_messages = messages[:2]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process images
        image_inputs, video_inputs = process_vision_info(input_messages)

        inputs = tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate with streaming
        print(f"\n🤖 MODEL OUTPUT (streaming):")
        from transformers import TextStreamer
        streamer = TextStreamer(tokenizer, skip_prompt=True,
                               skip_special_tokens=True)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.1,
                do_sample=False,
                streamer=streamer,
            )

        # Decode full output for comparison
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Compare
        print(f"\n{'─' * 70}")
        print(f"Example {i + 1}/{len(examples)}")
        print(f"{'─' * 70}")

        # Show image path
        user_content = messages[1]["content"]
        img_path = user_content[0].get("image", "?")
        print(f"Image: {img_path}")
        print(f"Prompt: {user_content[1]['text']}")

        print(f"\n📋 EXPECTED (first 300 chars):")
        print(expected[:300])

        # Try to parse and compare key fields
        try:
            expected_json = json.loads(expected)
            output_json = json.loads(output_text)

            # Compare structure_type
            exp_type = expected_json.get("structure_type", "")
            out_type = output_json.get("structure_type", "")
            type_match = exp_type == out_type

            # Compare sections
            exp_sections = set()
            out_sections = set()
            for key in ["columns", "beams", "bracing", "members"]:
                for m in expected_json.get(key, []):
                    if isinstance(m, dict) and "section" in m:
                        exp_sections.add(m["section"])
                for m in output_json.get(key, []):
                    if isinstance(m, dict) and "section" in m:
                        out_sections.add(m["section"])

            sections_match = exp_sections == out_sections

            print(f"\n✅ Valid JSON: yes")
            print(f"   Structure type: {'✅' if type_match else '❌'} "
                  f"(expected={exp_type}, got={out_type})")
            print(f"   Sections match: {'✅' if sections_match else '❌'} "
                  f"(expected={exp_sections}, got={out_sections})")

            if type_match and sections_match:
                correct += 1
            total += 1

        except json.JSONDecodeError:
            print(f"\n❌ Model output is NOT valid JSON")
            total += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{total} examples with correct type + sections")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
