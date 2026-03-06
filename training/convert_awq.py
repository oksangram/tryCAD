"""
Convert fine-tuned LoRA adapter to merged AWQ 4-bit model for vLLM serving.

Two-step process:
1. Merge LoRA adapter with base model → full 16-bit model
2. Quantize to AWQ 4-bit → ~40GB model for efficient vLLM serving

Run on RunPod H100 80GB (needs the full model in memory for merging):
  python training/convert_awq.py --adapter models/ift_checkpoint
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA + AWQ quantization")
    parser.add_argument("--adapter", default="models/ift_checkpoint",
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output-merged", default="models/merged_16bit",
                        help="Output for merged 16-bit model")
    parser.add_argument("--output-awq", default="models/structural_dsl_awq",
                        help="Output for AWQ quantized model")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HuggingFace repo to push AWQ model (optional)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (already have merged model)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("STRUCTURAL DSL — MODEL CONVERSION")
    print("=" * 60)
    print(f"  Adapter: {args.adapter}")
    print(f"  Merged output: {args.output_merged}")
    print(f"  AWQ output: {args.output_awq}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Config validated. Exiting.")
        return

    # ── Step 1: Merge LoRA adapter ──
    if not args.skip_merge:
        print("\n[Step 1/2] Merging LoRA adapter with base model...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=8192,
            dtype=None,
            load_in_4bit=False,  # Load in full precision for merging
        )

        # Save merged model in 16-bit
        os.makedirs(args.output_merged, exist_ok=True)
        model.save_pretrained_merged(
            args.output_merged,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"  Merged model saved to: {args.output_merged}")

        # Free memory
        del model
        import torch
        torch.cuda.empty_cache()
    else:
        print("\n[Step 1/2] Skipping merge (using existing merged model).")

    # ── Step 2: AWQ quantization ──
    print("\n[Step 2/2] Quantizing to AWQ 4-bit...")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.output_merged)
        model = AutoAWQForCausalLM.from_pretrained(
            args.output_merged,
            device_map="auto",
        )

        # AWQ quantization config
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }

        # Calibration data — use some of our training scripts
        calib_data = _load_calibration_data()

        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_data,
        )

        os.makedirs(args.output_awq, exist_ok=True)
        model.save_quantized(args.output_awq)
        tokenizer.save_pretrained(args.output_awq)

        print(f"  AWQ model saved to: {args.output_awq}")

        if args.push_to_hub:
            print(f"\n  Pushing to HuggingFace Hub: {args.push_to_hub}")
            model.push_to_hub(args.push_to_hub)
            tokenizer.push_to_hub(args.push_to_hub)

    except ImportError:
        print("  autoawq not installed. Install with: pip install autoawq")
        print("  Alternatively, use Unsloth's built-in GGUF export:")
        _export_gguf_fallback(args)


def _load_calibration_data(n_samples: int = 128) -> list[str]:
    """Load calibration data for AWQ quantization."""
    calib_texts = []

    # Use CPT corpus scripts for calibration
    cpt_path = Path("data/cpt_corpus.txt")
    if cpt_path.exists():
        text = cpt_path.read_text(encoding="utf-8")
        scripts = text.strip().split("\n\n")
        for script in scripts[:n_samples]:
            if len(script.strip()) > 50:
                calib_texts.append(script.strip())

    # Pad with IFT examples if needed
    if len(calib_texts) < n_samples:
        ift_path = Path("data/ift_train.jsonl")
        if ift_path.exists():
            with open(ift_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(calib_texts) >= n_samples:
                        break
                    ex = json.loads(line)
                    for msg in ex.get("messages", []):
                        if msg.get("content"):
                            calib_texts.append(msg["content"])
                            if len(calib_texts) >= n_samples:
                                break

    return calib_texts[:n_samples]


def _export_gguf_fallback(args):
    """Fallback: export as GGUF Q4_K_M using Unsloth."""
    print("\n  Using Unsloth GGUF export as fallback...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=8192,
    )

    gguf_output = args.output_awq.replace("awq", "gguf")
    os.makedirs(gguf_output, exist_ok=True)

    model.save_pretrained_gguf(
        gguf_output,
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"  GGUF model saved to: {gguf_output}")


if __name__ == "__main__":
    main()
