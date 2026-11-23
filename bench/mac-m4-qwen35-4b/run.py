"""Drive one traced llama-server config through profile_exact_sweep.py (the vLLM harness, unchanged).

python run.py <name> <cells: full|screen> [-- extra llama-server flags]

Env:
  MAC_HOST   ssh host of the Mac (default rentamac); "local" runs everything on this machine (no ssh, no tunnel).
  MAC_REPO   this repo's path on the Mac (default ~/llama.cpp-mac-m4-server-bench).
  MODEL      GGUF path on the Mac (default ~/models/ggml-org-Qwen3.5-4B-GGUF/Qwen3.5-4B-BF16.gguf).
  METAL_ENV  extra env for llama-server, e.g. "GGML_METAL_TENSOR_ENABLE=1".
Results land in results/<name>/ (CSV, traces, generated text, server log).
"""
import json, os, shlex, subprocess, sys, time
from pathlib import Path
import requests

root = Path(__file__).resolve().parent
client = root / "client"
name, cells = sys.argv[1], sys.argv[2]
flags = sys.argv[4:] if len(sys.argv) > 3 and sys.argv[3] == "--" else []
lengths, sizes = (["128", "512", "1024"], ["1", "2", "4"]) if cells == "full" else (["1024"], ["1", "4"])
host = os.environ.get("MAC_HOST", "rentamac")
mac_repo = os.environ.get("MAC_REPO", "~/llama.cpp-mac-m4-server-bench")
local = host == "local"
server_port, client_port = 18080, (18080 if local else 18003)
base = f"http://127.0.0.1:{client_port}"
remote_out = "~/llamacpp-server-bench"
tag = f"llamacpp_monitor_{name}"
out = root / "results" / name; out.mkdir(parents=True, exist_ok=True)
records = [json.loads(s) for s in (client / "exact_prompts.jsonl").read_text().splitlines()]


def on_mac(command, **kw):
    return subprocess.run(["sh", "-c", command] if local else ["ssh", host, command], **kw)


busy = on_mac('pgrep -fl "python|cmake|pytest|engine_bench|ninja|make|clang|llama-server|vllm|mlx" | grep -v -E "pgrep|run.py"',
              capture_output=True, text=True).stdout.strip()
if busy:
    sys.exit(f"Mac busy:\n{busy}")
tunnel = None if local else subprocess.Popen(["ssh", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-N",
                                              "-L", f"{client_port}:127.0.0.1:{server_port}", host])
# logs foreign processes that start during the run (anything here invalidates the numbers)
monitor_cmd = (f"while true; do : {tag}; ps -eo pid,lstart,command | grep -E 'python|pytest|mlx|vllm|engine_bench' "
               "| grep -v -E 'grep|resource_tracker|run.py' | sed \"s/^/$(date +%H:%M:%S) /\"; sleep 5; done")
monitor = subprocess.Popen(["sh", "-c", monitor_cmd] if local else ["ssh", host, monitor_cmd], stdout=(out / "foreign.log").open("w"))
try:
    metal_env = os.environ.get("METAL_ENV", "")
    model = f"MODEL={shlex.quote(os.environ['MODEL'])} " if "MODEL" in os.environ else ""
    on_mac(f"cd {mac_repo} && {model}PORT={server_port} BENCH_OUT={remote_out} METAL_ENV='{metal_env}' "
           f"sh bench/mac-m4-qwen35-4b/start_server.sh {name} {' '.join(flags)}", check=True)
    try:
        for _ in range(300):
            try:
                if requests.get(base + "/health", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("server startup timeout")
        payload = {k: v for k, v in records[4].items() if k != "metadata"}
        payload.update(max_tokens=16, temperature=0)
        warm = requests.post(base + "/v1/completions", json=payload, timeout=600)
        warm.raise_for_status()
        (out / "warmup.json").write_text(json.dumps(warm.json(), indent=2))
        env = dict(os.environ, PYTHONPATH=str(client / "score_tracing_shim"))
        command = [sys.executable, str(client / "profile_exact_sweep.py"), "--base-url", base, "--mode", "chat",
                   "--dataset", str(client / "exact_prompts.jsonl"), "--prompt-lengths", *lengths, "--batch-sizes", *sizes,
                   "--model", records[0]["model"], "--max-completion-tokens", "128",
                   "--output-csv", str(out / "profile_sweep.csv"), "--trace-dir", str(out / "traces"),
                   "--timeout", "1800", "--save-outputs", "--reset"]
        (out / "command.json").write_text(json.dumps({"client": command, "server_flags": flags, "metal_env": metal_env}, indent=2))
        subprocess.run(command, env=env, check=True)
    finally:
        on_mac(f'kill "$(cat {remote_out}/{name}/server.pid)"', check=False)
        time.sleep(2)
        src = os.path.expanduser(f"{remote_out}/{name}/") if local else f"{host}:{remote_out.replace('~/', '')}/{name}/"
        subprocess.run(["rsync", "-a", src, str(out) + "/"], check=False)
finally:
    monitor.terminate()
    if tunnel:
        tunnel.terminate()
    on_mac(f"pkill -f {tag}", check=False)
