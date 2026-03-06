"""
Continued Pre-Training (CPT) with Unsloth + QLoRA.

Teaches Qwen2.5-72B the STAAD Pro DSL syntax and structural patterns
before fine-tuning on tool-call conversations.

Run on RunPod H100 80GB:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps xformers trl peft accelerate bitsandbytes
  python training/train_cpt.py
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CPT training with Unsloth QLoRA")
    parser.add_argument("--model", default="unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
                        help="Base model (Unsloth 4-bit pre-quantized)")
    parser.add_argument("--data", default="data/cpt_train.jsonl",
                        help="CPT training data JSONL")
    parser.add_argument("--output", default="models/cpt_checkpoint",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--save-steps", type=int, default=200,
                        help="Save checkpoint every X steps (for spot instances)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without training")
    args = parser.parse_args()

    config = {
        "model": args.model,
        "data": args.data,
        "output": args.output,
        "max_seq_len": args.max_seq_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
    }

    print("=" * 60)
    print("STRUCTURAL DSL — CONTINUED PRE-TRAINING")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Config validated. Exiting.")
        return config

    # ── Load model with Unsloth ──
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,       # Auto-detect (bf16 on H100)
        load_in_4bit=True,
    )

    # ── Apply LoRA ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Load dataset ──
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"\nDataset: {len(dataset)} documents")

    # Tokenize for CLM (causal language modeling)
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # ── Train ──
    from trl import SFTTrainer
    from transformers import TrainingArguments

    os.makedirs(args.output, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=True,  # Pack multiple short docs into one sequence
        args=training_args,
    )

    # Auto-resume from checkpoint if interrupted
    import glob
    checkpoints = glob.glob(os.path.join(args.output, "checkpoint-*"))
    resume = len(checkpoints) > 0

    print(f"\nStarting CPT training... (Resuming: {resume})")
    try:
        stats = trainer.train(resume_from_checkpoint=resume)
        print(f"\nTraining complete. Loss: {stats.training_loss:.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Model can be resumed later.")
        return config

    # ── Save ──
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to: {args.output}")

    # Save training config
    with open(os.path.join(args.output, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return config


if __name__ == "__main__":
    main()
