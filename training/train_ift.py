"""
Instruction Fine-Tuning (IFT) with Unsloth + QLoRA.

Fine-tunes the CPT checkpoint on chat-with-tools conversations,
teaching the model to reason about structures and call tools correctly.

Run on RunPod H100 80GB:
  python training/train_ift.py --base-model models/cpt_checkpoint
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="IFT training with Unsloth QLoRA")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
                        help="Base model ID or local checkpoint path")
    parser.add_argument("--train-data", default="data/ift_train_prepared.jsonl",
                        help="Training data JSONL")
    parser.add_argument("--eval-data", default="data/ift_eval_prepared.jsonl",
                        help="Evaluation data JSONL")
    parser.add_argument("--output", default="models/ift_checkpoint",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--max-seq-len", type=int, default=8192,
                        help="Maximum sequence length (longer for multi-turn tool calls)")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=2e-5,
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
        "base_model": args.base_model,
        "train_data": args.train_data,
        "eval_data": args.eval_data,
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
    print("STRUCTURAL DSL — INSTRUCTION FINE-TUNING")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Config validated. Exiting.")
        return config

    # ── Set cache dirs to persistent volume ──
    os.environ["HF_HOME"] = "/workspace/huggingface_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"
    os.environ["HF_HUB_CACHE"] = "/workspace/huggingface_cache"
    os.makedirs("/workspace/huggingface_cache", exist_ok=True)

    # ── Load model ──
    from unsloth import FastLanguageModel

    # Check if base is a Unsloth-quantized model or a local checkpoint
    is_local = os.path.isdir(args.base_model)

    print(f"\nDownloading/loading model: {args.base_model}")
    print(f"HF cache: {os.environ['HF_HOME']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    # If using fresh base model (no CPT), apply LoRA from scratch
    # If using CPT checkpoint, the LoRA weights are already loaded
    if not is_local:
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

    # ── Load datasets ──
    from datasets import load_dataset

    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = None
    if os.path.exists(args.eval_data):
        eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")

    print(f"\nTrain: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Eval:  {len(eval_dataset)} examples")

    # ── Formatting function for chat template ──
    def formatting_func(examples):
        """Convert messages list to Qwen2.5 chat template string."""
        texts = []
        for messages in examples["messages"]:
            # We strictly use the robust manual formatter because HF apply_chat_template
            # often throws Jinja UndefinedError exceptions on custom tool call dictionaries.
            text = _manual_format(messages)
            texts.append(text)
        return texts

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
        eval_strategy="epoch" if eval_dataset else "no",
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_len,
        packing=False,  # No packing for chat data — each example is one conversation
        args=training_args,
    )

    # Auto-resume from checkpoint if interrupted
    import glob
    checkpoints = glob.glob(os.path.join(args.output, "checkpoint-*"))
    resume = len(checkpoints) > 0

    print(f"\nStarting IFT training... (Resuming: {resume})")
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

    # Save config
    with open(os.path.join(args.output, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Eval metrics ──
    if eval_dataset:
        eval_results = trainer.evaluate()
        print(f"\nEval results: {eval_results}")
        with open(os.path.join(args.output, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

    return config


def _manual_format(messages: list) -> str:
    """Fallback manual formatting for messages with tool calls."""
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            tc = msg.get("tool_calls", [])
            if tc:
                tc_str = "\n".join(
                    f'<tool_call>\n{json.dumps({"name": t["function"]["name"], "arguments": json.loads(t["function"]["arguments"])})}\n</tool_call>'
                    for t in tc
                )
                parts.append(f"<|im_start|>assistant\n{content}\n{tc_str}<|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            tool_id = msg.get("tool_call_id", "")
            parts.append(f"<|im_start|>tool\n{content}<|im_end|>")

    return "\n".join(parts)


if __name__ == "__main__":
    main()
