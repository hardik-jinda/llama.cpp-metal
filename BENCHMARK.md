# llama.cpp server benchmark: Qwen3.5-4B BF16 on Mac mini M4 (Metal)

This repo is llama.cpp at upstream commit `44d50873b` (tree only, history not included) plus an opt-in
step-timing trace in `llama-server`, and the harness that benchmarked it on 2026-11-23 against vLLM-MLX and
Score Engine. The trace emits the same per-step records and `/debug/*` endpoints as the vLLM-Metal
`metal_timing` plugin, so the exact client used for vLLM drives `llama-server` unchanged.

Commits:

| commit | content |
|---|---|
| `import llama.cpp 44d50873b` | upstream tree, `https://github.com/ggml-org/llama.cpp` `44d50873b56b568c28ef2ad8d6445069feaa9fbd`, minus `.github/workflows` (upstream CI, not needed) |
| `server: add opt-in step timing trace ...` | `tools/server/server-trace.h` (new), hooks in `server-context.cpp`, endpoints in `server.cpp` |
| `bench: ...` | `bench/mac-m4-qwen35-4b/` (harness, prompts, raw results) and this file |

The same server change is also in `bench/mac-m4-qwen35-4b/llama-server-trace-timing.patch`, to apply to a
fresh upstream checkout (`git apply`; it may need rebasing on newer upstream).

## What was measured

- **Server path only**: HTTP `/v1/completions`, B concurrent requests, one server process.
- Cells: prompt length 128 / 512 / 1024 tokens x batch size 1 / 2 / 4.
- Each request: exact prompt token IDs (no tokenization by the server), 128 output tokens, `temperature 0`,
  `ignore_eos true`, `seed 0`. The client asserts the exact prompt length and 128 completion tokens.
- One 16-token warmup request after startup, excluded from all windows.
- Prompt cache off (`--no-cache-prompt --cache-ram 0`), so each cell is a cold prefill.

## Reference machine

| item | value |
|---|---|
| machine | Mac mini M4 (Mac16,10), 10-core GPU, 16 GB unified memory, ~120 GB/s |
| OS | macOS 26.5.2 (25F84), SDK 26.5 |
| compiler | Apple clang 21.0.0 (clang-2100.1.1.101) |
| cmake | 4.1.0 |
| build | `Release`, `GGML_METAL=ON`, `BUILD_SHARED_LIBS=ON`, `LLAMA_CURL=OFF` |
| client host | any machine with Python 3.12 + `requests` 2.31 (ran on Linux, reaching the Mac over an ssh tunnel) |

Nothing else ran on the Mac during the sweep; `run.py` refuses to start if it sees python, cmake, clang,
llama-server, vllm or mlx processes, and logs any that appear mid-run to `results/<name>/foreign.log`.

## Reproduce

### 1. Mac: clone and build

```sh
git clone https://github.com/hardik-jinda/llama.cpp-mac-m4-server-bench.git ~/llama.cpp-mac-m4-server-bench  # private repo: needs access
cd ~/llama.cpp-mac-m4-server-bench
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j 8
ls build/bin/llama-server
```

Needs Xcode command line tools and cmake >= 3.14 (`brew install cmake`, or any cmake on PATH).

### 2. Mac: weights

Official GGUF from ggml-org, BF16 (no quantization, same precision as the vLLM and Score runs):

```sh
mkdir -p ~/models/ggml-org-Qwen3.5-4B-GGUF && cd ~/models/ggml-org-Qwen3.5-4B-GGUF
curl -L -o Qwen3.5-4B-BF16.gguf https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-BF16.gguf
shasum -a 256 Qwen3.5-4B-BF16.gguf
# 23ac53ca9a982b6e5f400a8d0efc97cf70734192e47ecd4babd8e7e0a53ed579  (8424393664 bytes)
```

Another path works via `MODEL=/path/to.gguf`.

### 3. Client: run the sweep

On the client (this repo cloned there too, `pip install requests`):

```sh
cd bench/mac-m4-qwen35-4b
# MAC_HOST: ssh alias of the Mac (passwordless ssh); MAC_REPO: repo path on the Mac
MAC_HOST=rentamac MAC_REPO='~/llama.cpp-mac-m4-server-bench' python3 run.py kvu_r1 full -- -fa on -ctxcp 0 -kvu
MAC_HOST=rentamac MAC_REPO='~/llama.cpp-mac-m4-server-bench' python3 run.py kvu_r2 full -- -fa on -ctxcp 0 -kvu
python3 summarize.py results/kvu_r1 results/kvu_r2
```

To run everything on the Mac itself, use `MAC_HOST=local` (no ssh, no tunnel; server on port 18080).

`run.py` per config: busy check, ssh tunnel `127.0.0.1:18003 -> mac:18080`, starts `start_server.sh` on the
Mac (`LLAMA_SERVER_TRACE_TIMING=1`, base flags `-ngl 99 -np 4 -c 8192 --no-cache-prompt --cache-ram 0` plus the
flags after `--`), waits for `/health`, sends the warmup, runs `client/profile_exact_sweep.py`, kills the
server and rsyncs its log back. A full 9-cell sweep takes a few minutes; `screen` mode runs only
1x1024 and 4x1024.

Flag screening (`screen.sh`) re-runs the s0..s8 configs in `results/`.

### Outputs per run (`results/<name>/`)

- `profile_sweep.csv`: one row per cell. Same schema as the vLLM CSV (`batch_wall_ms`, `prefill_*`,
  `decode_forward_p50_ms`, `decode_full_batch_steps`, ...).
- `traces/trace_len<L>_bs<B>.json`: raw step records. `traces/outputs_*.jsonl`: generated text.
- `server.log`, `flags.txt`, `metal_env.txt`, `commit.txt`, `command.json`, `warmup.json`.

Always read the generated text, not just the CSV status. In the reference run it was coherent and matched
vLLM's text word for word in the rows checked.

## Metric definitions

- **Step** (the trace): one `llama_process()` call of `update_slots()` plus the sampling after it. When
  tracing is on, the server calls `llama_synchronize()` after every step. llama-server already syncs on
  every step that samples, so the extra cost is close to zero. `forward_ms` = step start -> GPU done.
- **Prefill tok/s** = `B * prompt_len / (first step start -> end of the step that completes the last
  prompt)` (`summarize.py`). This is the method that reproduces the corrected vLLM numbers exactly. The CSV
  `prefill_*` columns are per-request accumulated forward time, kept for schema parity.
- **Decode p50 ms**: median `forward_ms` over full-batch pure-decode steps (all B requests decoding).
- **Wall s**: client-side `batch_wall_ms`, first request sent -> last response received.

## Flag findings (1x1024 and 4x1024)

| flag | effect | kept |
|---|---|---|
| `-ctxcp 0` | +3% prefill. Default context checkpoints split each prompt into 3 steps (end-4-ubatch, end-4) to snapshot recurrent state, useless with prompt cache off | yes |
| `-kvu` (unified KV) | bs2 decode 163 -> 90 ms when the two requests land on non-adjacent slots; without it the speed depends on which slots get picked (`full_r1`/`full_r2` show it) | yes |
| `-fa on` vs `off` | noise | `on` |
| `-b 4096 -ub 1024/2048/4096` | noise | default `-b 2048 -ub 512` |
| `GGML_METAL_TENSOR_ENABLE=1` | prefill -10% | no |

`ggml-metal-tuning` (tunes only the decode flash-attention vec kernel, takes hours) was not run.

Final flags: `-ngl 99 -np 4 -c 8192 --no-cache-prompt --cache-ram 0 -fa on -ctxcp 0 -kvu`.

## Reference results (`results/kvu_r1`, `kvu_r2` within ~1%)

| B x prompt | prefill tok/s | decode p50 ms | wall s |
|---|---|---|---|
| 1 x 128 | 391.4 | 81.7 | 10.93 |
| 1 x 512 | 421.4 | 81.4 | 11.76 |
| 1 x 1024 | 422.9 | 81.5 | 12.99 |
| 2 x 128 | 352.2 | 90.0 | 12.38 |
| 2 x 512 | 406.2 | 89.9 | 14.16 |
| 2 x 1024 | 414.6 | 90.2 | 16.64 |
| 4 x 128 | 385.4 | 96.1 | 13.80 |
| 4 x 512 | 409.7 | 96.3 | 17.49 |
| 4 x 1024 | 413.5 | 97.1 | 22.49 |

A rerun on the same hardware should land within a few percent. Different chips (M4 Pro/Max, M3) shift
absolute numbers; compare ratios against vLLM/Score run on the same machine instead.

### Comparison CSVs (`bench/mac-m4-qwen35-4b/results/`)

- `llamacpp_server_profile_sweep_20261123.csv`, `..._repeat.csv`: copies of `kvu_r1`/`kvu_r2` CSVs.
- `vllm_mlx_profile_sweep_20261109.csv`: vLLM-MLX on the same Mac, same client and prompts (2026-11-09).
  Its 4x1024 cell is flawed by a re-prefill.
- `13_score_vs_llamacpp_vs_vllm.csv`: Score Engine vs llama.cpp vs vLLM. Score numbers come from in-process
  `engine_bench` (Score repo, branch `mac-gpu` `74bf0df6`), not HTTP; Score's prompts are 1-9% longer
  (chat template); Score's first cell (1x128) carries ~1.8 s of one-time Metal shader compiles.

## Client inputs (`bench/mac-m4-qwen35-4b/client/`)

- `exact_prompts.jsonl`: 12 prompts (128/512/1024 tokens x 4 War and Peace slices), token IDs from the
  Qwen3.5-4B HF tokenizer + chat template, trimmed in the middle to the exact length. Generated by
  `prepare_prompts.py` (needs the source `original_prompts.jsonl`, not included; the saved IDs are the input).
  The `model` field is a path on the original Mac; llama-server ignores it.
- `profile_exact_sweep.py`: the vLLM client, unchanged. `score_tracing_shim/` provides the one helper it
  imports (`score_engine.tracing.collect_trace_pages`, trace pagination) so it runs without Score installed.

## Porting the trace to a newer llama.cpp

`server-trace.h` is self-contained. The hooks in `server-context.cpp` sit in `update_slots()`: take
`t_start_us` before the batch is built, `llama_synchronize()` + `t_fwd_end_us` after `llama_process()`,
classify slots (prompt tokens in batch / prompt done / generating) before `post_decode()`, record after it.
Sampling and detokenize timers wrap `common_sampler_sample` and `common_token_to_piece`. If upstream moves
these, re-place the hooks by hand and check a trace: every step needs `prefills`/`decode` filled, and
`decode_full_batch_steps` should be 126-127 per cell.
