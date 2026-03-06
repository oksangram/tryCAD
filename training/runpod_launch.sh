#!/usr/bin/env bash
#
# RunPod Training Launch Script
#
# Upload this to a RunPod H100 pod, then run:
#   chmod +x training/runpod_launch.sh
#   ./training/runpod_launch.sh
#
# Prerequisites:
#   - RunPod H100 80GB pod with PyTorch 2.1+ and CUDA 12.1+
#   - Upload project files to /workspace/tryCAD_AI/
#
set -euo pipefail

echo "=============================================="
echo "  STRUCTURAL DSL — TRAINING PIPELINE"
echo "=============================================="

cd /workspace/tryCAD_AI

# ── 1. Install dependencies ──
echo ""
echo "[1/6] Installing dependencies..."
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps xformers trl peft accelerate bitsandbytes
pip install -q datasets lark autoawq
pip install -q -r requirements.txt 2>/dev/null || true

# ── 2. Generate training data (if not already generated) ──
if [ ! -f data/ift_train.jsonl ] || [ $(wc -l < data/ift_train.jsonl) -lt 1000 ]; then
    echo ""
    echo "[2/6] Generating training data (~14000 target)..."
    python -m datagen.pipeline 14000
else
    echo ""
    echo "[2/6] Training data already exists ($(wc -l < data/ift_train.jsonl) examples). Skipping."
fi

# ── 3. Prepare CPT data ──
echo ""
echo "[3/6] Preparing CPT data..."
python training/prepare_cpt_data.py

# ── 4. Prepare IFT data ──
echo ""
echo "[4/6] Preparing IFT data..."
python training/prepare_ift_data.py

# ── 5. CPT Training (~4-6 hours) ──
echo ""
echo "[5/6] Starting Continued Pre-Training..."
echo "       Estimated time: 4-6 hours on H100"
python training/train_cpt.py \
    --model unsloth/Qwen2.5-72B-Instruct-bnb-4bit \
    --data data/cpt_train.jsonl \
    --output models/cpt_checkpoint \
    --max-seq-len 4096 \
    --lora-r 64 \
    --lora-alpha 128 \
    --lr 5e-5 \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8

# ── 6. IFT Training (~4-6 hours) ──
echo ""
echo "[6/6] Starting Instruction Fine-Tuning..."
echo "       Estimated time: 4-6 hours on H100"
python training/train_ift.py \
    --base-model models/cpt_checkpoint \
    --train-data data/ift_train_prepared.jsonl \
    --eval-data data/ift_eval_prepared.jsonl \
    --output models/ift_checkpoint \
    --max-seq-len 8192 \
    --lora-r 64 \
    --lora-alpha 128 \
    --lr 2e-5 \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8

echo ""
echo "=============================================="
echo "  TRAINING COMPLETE"
echo "=============================================="
echo ""
echo "Checkpoints:"
echo "  CPT: models/cpt_checkpoint/"
echo "  IFT: models/ift_checkpoint/"
echo ""
echo "Next: Run AWQ conversion:"
echo "  python training/convert_awq.py --adapter models/ift_checkpoint"
