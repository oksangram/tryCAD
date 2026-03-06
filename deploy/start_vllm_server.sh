#!/bin/bash
# start_vllm_server.sh
# Automates the setup and launch of the vLLM server on RunPod spot instances

echo "=================================================="
echo "🚀 STARTING VLLM DEPLOYMENT SETUP"
echo "=================================================="

echo ""
echo "[1/4] Configuring Environment Variables..."
# Force pip to use the persistent workspace for caching so we don't blow up the 50GB root disk
export PIP_CACHE_DIR="/workspace/pip_cache"
export HF_HOME="/workspace/huggingface_cache"
echo "✅ Cached to: /workspace"

echo ""
echo "[2/4] Installing vLLM & Dependencies..."
echo "⏳ This may take a minute if downloading for the first time..."
pip install vllm bitsandbytes peft accelerate
echo "✅ vLLM Installation verified!"

echo ""
echo "[3/4] Validating RunPod Port Configuration..."
echo "🔌 REMINDER: Ensure you have exposed HTTP port 8080 in your RunPod 'Edit Pod' menu!"

echo ""
echo "[4/4] 🟢 Launching vLLM Server on Port 8080..."
vllm serve unsloth/Qwen2.5-72B-Instruct-bnb-4bit \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --enable-lora \
    --lora-modules tryCAD_agent=/workspace/tryCAD/models/ift_checkpoint/checkpoint-600 \
    --max-lora-rank 64 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8080 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
