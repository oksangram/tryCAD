FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    vllm==0.4.3 \
    autoawq \
    lark>=1.1.0 \
    runpod

# Copy project files
COPY dsl/ /app/dsl/
COPY tools/ /app/tools/
COPY validation/ /app/validation/
COPY standards/ /app/standards/
COPY deploy/ /app/deploy/

# Model will be mounted or downloaded at runtime
ENV MODEL_PATH=/models/structural_dsl_awq
ENV MAX_TOOL_ROUNDS=10
ENV MAX_TOKENS=4096
ENV PYTHONPATH=/app

# RunPod serverless entrypoint
CMD ["python", "-u", "deploy/runpod_handler.py"]
