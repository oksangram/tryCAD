"""
Fine-tune Qwen3-VL-8B with Unsloth QLoRA for structural drawing extraction.

Run on RunPod H100 80GB or similar GPU.

Prerequisites (install on RunPod):
    pip install unsloth
    pip install --no-deps trl peft accelerate bitsandbytes

Usage:
    python extraction_training/train_extraction.py \
        --train-data data/extraction_vl_train.jsonl \
        --eval-data data/extraction_vl_train_eval.jsonl \
        --output-dir models/extraction_vl_lora \
        --epochs 3
"""

from __future__ import annotations
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL-8B for structural drawing extraction")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to training JSONL (from prepare_extraction_data.py)")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Path to eval JSONL (optional)")
    parser.add_argument("--output-dir", type=str, default="models/extraction_vl_lora",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Qwen3-VL-8B-Instruct",
                        help="Base model ID")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and data loading without training")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-VL-8B Extraction Fine-tuning")
    print("=" * 60)
    print(f"  Base model:    {args.base_model}")
    print(f"  Train data:    {args.train_data}")
    print(f"  Eval data:     {args.eval_data or 'none'}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank:     {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  Max seq len:   {args.max_seq_length}")
    print()

    # ── Step 1: Load base model with Unsloth ──
    print("[1/5] Loading base model with Unsloth 4-bit quantization...")

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Tokenizer: {type(tokenizer).__name__}")

    # ── Step 2: Attach LoRA adapter ──
    print(f"\n[2/5] Attaching LoRA adapter (rank={args.lora_rank})...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ── Step 3: Load and format dataset ──
    print(f"\n[3/5] Loading training data from {args.train_data}...")

    def load_jsonl(path):
        examples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    train_raw = load_jsonl(args.train_data)
    print(f"  Loaded {len(train_raw)} training examples")

    # Use plain PyTorch Dataset — avoids pyarrow mixed-type errors
    # (user content is a list, system/assistant content is a string)
    from torch.utils.data import Dataset as TorchDataset

    class ChatDataset(TorchDataset):
        """Simple wrapper — avoids pyarrow schema issues with mixed-type content."""
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = ChatDataset(train_raw)

    eval_dataset = None
    if args.eval_data and os.path.exists(args.eval_data):
        eval_raw = load_jsonl(args.eval_data)
        eval_dataset = ChatDataset(eval_raw)
        print(f"  Loaded {len(eval_raw)} eval examples")

    if args.dry_run:
        print("\n[DRY RUN] Validating first example...")
        ex = train_raw[0]
        print(f"  System: {ex['messages'][0]['content'][:80]}...")
        user_msg = ex['messages'][1]['content']
        print(f"  User image: {user_msg[0]['image'][:60]}...")
        print(f"  User text: {user_msg[1]['text'][:80]}...")
        assistant = ex['messages'][2]['content']
        print(f"  Assistant JSON length: {len(assistant)} chars")
        # Validate JSON
        parsed = json.loads(assistant)
        print(f"  Assistant JSON keys: {list(parsed.keys())}")
        print("\n✅ Dry run passed. Config and data look valid.")
        return

    # ── Step 4: Configure trainer ──
    print(f"\n[4/5] Configuring SFT trainer...")

    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported, UnslothVisionDataCollator

    FastVisionModel.for_training(model)

    # Compute warmup steps to avoid deprecated warmup_ratio
    total_steps = (len(train_dataset) * args.epochs) // (
        args.batch_size * args.grad_accum)
    warmup_steps = max(1, int(total_steps * 0.1))

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        max_seq_length=args.max_seq_length,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
    )

    print(f"  Total training steps: "
          f"{len(train_dataset) * args.epochs // (args.batch_size * args.grad_accum)}")

    # ── Step 5: Train ──
    print(f"\n[5/5] Starting training...")
    print("=" * 60)

    trainer_stats = trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Total time: {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"  Train loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # ── Save ──
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also save merged model for easier inference
    merged_dir = args.output_dir + "_merged"
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"\n✅ Done! LoRA adapter: {args.output_dir}")
    print(f"   Merged model: {merged_dir}")


if __name__ == "__main__":
    main()
