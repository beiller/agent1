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
    --parallel 1 \
    --host 0.0.0.0 --port 8080
