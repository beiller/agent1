#!/bin/bash
# Start llama-server in ROUTER mode (no --model flag)
# Models are loaded dynamically via API

/home/bill/src/llama.cpp/build/bin/llama-server \
    --models-dir /home/bill/Downloads \
    --ctx-size 65535 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 --jinja -t 8 --n-gpu-layers 99 \
    --flash-attn on --cache-reuse 512 --fit on -b 4096 -ub 1024 --parallel 1 \
    --host 0.0.0.0 --port 8080

