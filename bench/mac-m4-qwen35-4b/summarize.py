"""Per-cell summary of profile_sweep.csv runs: prefill tok/s from the trace window (the corrected vLLM method), decode p50, batch wall."""
import csv, json, sys
from pathlib import Path

def summarize(run):
    out = []
    for r in csv.DictReader(open(Path(run) / "profile_sweep.csv")):
        bs, L = int(r["batch_size"]), int(r["prompt_len"])
        st = json.load(open(r["trace_json"]))["steps"]
        last = max(i for i, s in enumerate(st) if s["prefills"])
        window_ms = (st[last]["mono_ns_last"] - st[0]["mono_ns_first"]) / 1e6
        texts = [json.loads(l)["output_text"] for l in open(r["outputs_jsonl"])]
        out.append(dict(bs=bs, ctx=L, prefill_window_ms=round(window_ms, 1), prefill_tok_s=round(bs * L / window_ms * 1e3, 1),
                        prefill_steps=last + 1, decode_p50_ms=float(r["decode_forward_p50_ms"]), decode_mean_ms=float(r["decode_forward_mean_ms"]),
                        decode_full_batch_steps=int(r["decode_full_batch_steps"]), batch_wall_s=round(float(r["batch_wall_ms"]) / 1e3, 3),
                        text0=texts[0][:60]))
    return out

for run in sys.argv[1:]:
    for row in summarize(run):
        print(run, {k: v for k, v in row.items() if k != "text0"})
