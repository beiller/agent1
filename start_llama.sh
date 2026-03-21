#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env.example ]; then
    source .env.example
fi

if [ -f .env ]; then
    source .env
fi

# Default paths (override with .env or command line)
LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-./llama-server}"
LLAMA_MODELS_DIR="${LLAMA_MODELS_DIR:-./models}"

"$LLAMA_SERVER_PATH" \
    --models-dir "$LLAMA_MODELS_DIR" \
    --ctx-size 65535 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 --jinja -t 8 --n-gpu-layers 99 \
    --flash-attn on --cache-reuse 512 --fit on -b 4096 -ub 1024 --parallel 1 \
    --host 0.0.0.0 --port 8080
