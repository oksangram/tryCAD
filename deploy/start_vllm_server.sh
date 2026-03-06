#!/bin/bash
# start_vllm_server.sh
# Automates the setup and launch of the vLLM server on RunPod spot instances

echo "🚀 Starting vLLM Deployment Setup..."

# Force pip to use the persistent workspace for caching so we don't blow up the 50GB root disk
export PIP_CACHE_DIR="/workspace/pip_cache"
export HF_HOME="/workspace/huggingface_cache"

echo "📦 Installing vLLM (this will use the workspace cache)..."
pip install -q vllm

echo "🟢 Launching vLLM Server on Port 8888..."
# Launch on port 8888 so it is automatically exposed via RunPod's HTTP Proxy
vllm serve unsloth/Qwen2.5-72B-Instruct-bnb-4bit \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --enable-lora \
    --lora-modules tryCAD_agent=/workspace/tryCAD/models/ift_checkpoint/checkpoint-600 \
    --max-lora-rank 64 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8888 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
