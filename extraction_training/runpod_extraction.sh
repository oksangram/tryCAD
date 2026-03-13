#!/bin/bash
# ==============================================================================
# RunPod Launch Script — Qwen3-VL-8B Extraction Fine-tuning
# ==============================================================================
#
# This script runs on a RunPod H100 pod. It:
#   1. Installs dependencies (Unsloth, etc.)
#   2. Generates the full 3,500-image training dataset
#   3. Formats for Qwen3-VL training
#   4. Fine-tunes with QLoRA
#   5. Saves the merged model to /workspace
#
# Usage:
#   Upload this script and the extraction_training/ folder to RunPod /workspace
#   then: bash runpod_extraction.sh
#
# Estimated: ~4-5 hours total, ~$12-15 on H100
# ==============================================================================

set -e

echo "========================================"
echo "Qwen3-VL-8B Extraction Fine-tuning"
echo "========================================"
echo "Start time: $(date)"
echo ""

# ── Environment ──
export PIP_CACHE_DIR=/workspace/pip_cache
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/huggingface_cache

WORKSPACE=/workspace
PROJECT_DIR=$WORKSPACE/tryCAD

cd $PROJECT_DIR

# ── Step 1: Install dependencies ──
echo ""
echo "[1/5] Installing dependencies..."
echo "========================================"
pip install -q unsloth
pip install -q --no-deps trl peft accelerate bitsandbytes
pip install -q matplotlib Pillow datasets qwen-vl-utils
echo "✅ Dependencies installed"

# ── Step 2: Generate 3D multi-view training dataset ──
echo ""
echo "[2/5] Generating 3,500 Level 3 (3D multi-view) images..."
echo "========================================"

python -m extraction_training.synthetic_drawings \
    --level 3 \
    --count 3500 \
    --output data/extraction

echo "✅ 3,500 multi-view images generated"

# ── Step 3: Format for Qwen3-VL ──
echo ""
echo "[3/5] Formatting training data for Qwen3-VL..."
echo "========================================"

python -m extraction_training.prepare_extraction_data \
    --input data/extraction \
    --output data/extraction_vl_train.jsonl \
    --eval-split 0.1

echo "✅ Training data formatted"

# ── Step 4: Dry run validation ──
echo ""
echo "[4/5] Validating config (dry run)..."
echo "========================================"

python extraction_training/train_extraction.py \
    --train-data data/extraction_vl_train.jsonl \
    --eval-data data/extraction_vl_train_eval.jsonl \
    --output-dir $WORKSPACE/models/extraction_vl_lora \
    --dry-run

echo "✅ Dry run passed"

# ── Step 5: Full training ──
echo ""
echo "[5/5] Starting QLoRA fine-tuning..."
echo "========================================"
echo "  1 epoch, eval every 50 steps, save every 100 steps"
echo ""

python extraction_training/train_extraction.py \
    --train-data data/extraction_vl_train.jsonl \
    --eval-data data/extraction_vl_train_eval.jsonl \
    --output-dir $WORKSPACE/models/extraction_vl_lora \
    --base-model unsloth/Qwen3-VL-8B-Instruct \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 2e-5 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --max-seq-length 4096

echo ""
echo "========================================"
echo "✅ TRAINING COMPLETE"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "LoRA adapter: $WORKSPACE/models/extraction_vl_lora"
echo "Merged model: $WORKSPACE/models/extraction_vl_lora_merged"
echo ""
echo "Next: Download the merged model or deploy with vLLM"
