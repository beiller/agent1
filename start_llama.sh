


#MODEL="/home/bill/Downloads/Qwen_Qwen3.5-35B-A3B-Q4_K_L.gguf"
MODEL="/home/bill/Downloads/Qwen_Qwen3.5-27B-Q4_1.gguf"
#MODEL="/home/bill/Downloads/Qwen3.5-9B-ultra-heretic-Q8_0.gguf"
/home/bill/src/llama.cpp/build/bin/llama-server \
    --model ${MODEL} \
    --ctx-size 65535 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 --jinja -t 8 --n-gpu-layers 99 \
    --flash-attn on --cache-reuse 512 --fit on -b 4096 -ub 1024 --parallel 1


#./llama.cpp/build/bin/llama-server --model "/Users/bill/Downloads/Qwen_Qwen3.5-27B-IQ4_XS.gguf" -t 16
