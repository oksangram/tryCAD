#!/bin/bash
# ==============================================================================
# RunPod Startup Script — Install dependencies
# ==============================================================================
# Run this after each pod restart:
#   bash extraction_training/runpod_setup.sh
# ==============================================================================

set -e

export PIP_CACHE_DIR=/workspace/pip_cache
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/huggingface_cache

echo "Installing dependencies (cached — should be fast)..."
pip install -q unsloth
pip install -q --no-deps trl peft accelerate bitsandbytes
pip install -q matplotlib Pillow datasets qwen-vl-utils
echo "✅ All dependencies installed"
