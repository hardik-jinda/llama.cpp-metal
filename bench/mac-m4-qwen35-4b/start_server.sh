#!/bin/sh
# start_server.sh <name> [extra llama-server flags...]: traced llama-server on 127.0.0.1:$PORT, run on the Mac.
# Logs go to $BENCH_OUT/<name>/. Env: MODEL, PORT, BENCH_OUT, METAL_ENV="GGML_METAL_TENSOR_ENABLE=1 ..." (optional).
set -eu
name="$1"; shift
repo="$(cd "$(dirname "$0")/../.." && pwd)"
model="${MODEL:-$HOME/models/ggml-org-Qwen3.5-4B-GGUF/Qwen3.5-4B-BF16.gguf}"
port="${PORT:-18080}"
if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then echo "port $port busy" >&2; exit 3; fi
out="${BENCH_OUT:-$HOME/llamacpp-server-bench}/$name"; mkdir -p "$out"
cd "$repo"
echo "$*" > "$out/flags.txt"; echo "${METAL_ENV:-}" > "$out/metal_env.txt"; git rev-parse HEAD > "$out/commit.txt"
nohup env LLAMA_SERVER_TRACE_TIMING=1 ${METAL_ENV:-} ./build/bin/llama-server \
  -m "$model" --host 127.0.0.1 --port "$port" -ngl 99 -np 4 -c 8192 --no-cache-prompt --cache-ram 0 "$@" \
  > "$out/server.log" 2>&1 < /dev/null &
echo "$!" > "$out/server.pid"
