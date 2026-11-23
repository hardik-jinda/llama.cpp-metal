#!/usr/bin/env python3
"""Profile request sweeps from unified JSONL traces."""

import argparse
import concurrent.futures
import csv
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from score_engine.tracing import collect_trace_pages


MODE_CHAT = "chat"
MODE_CHAT_SKIP_PREFILL = "chat-skip-prefill"
MODE_GENERATE = "generate"
MODE_GENERATE_SKIP_PREFILL = "generate-skip-prefill"
SKIP_PREFILL_HEADER = "X-Score-Skip-Prefill"
CHAT_MODES = {MODE_CHAT, MODE_CHAT_SKIP_PREFILL}
GENERATE_MODES = {MODE_GENERATE, MODE_GENERATE_SKIP_PREFILL}
SKIP_PREFILL_MODES = {MODE_CHAT_SKIP_PREFILL, MODE_GENERATE_SKIP_PREFILL}

CSV_FIELDS = [
    "prompt_len",
    "batch_size",
    "max_completion_tokens",
    "start_step_id",
    "end_step_id",
    "trace_record_count",
    "finished_requests",
    "batch_wall_ms",
    "prefill_count",
    "prefill_min_ms",
    "prefill_max_ms",
    "prefill_mean_ms",
    "prefill_sampling_mean_ms",
    "decode_full_batch_steps",
    "decode_full_batch_tokens",
    "decode_full_batch_duration_ms",
    "decode_throughput",
    "decode_forward_min_ms",
    "decode_forward_max_ms",
    "decode_forward_mean_ms",
    "decode_forward_p50_ms",
    "decode_forward_p90_ms",
    "decode_sampling_mean_ms",
    "sampling_mean_ms",
    "tokenization_mean_ms",
    "detokenization_mean_ms",
    "trace_json",
    "outputs_jsonl",
]


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def request_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    if method == "POST":
        response = requests.post(
            url,
            json=payload or {},
            headers=headers,
            timeout=timeout,
        )
    elif method == "GET":
        response = requests.get(
            url,
            params=payload or {},
            headers=headers,
            timeout=timeout,
        )
    else:
        raise ValueError(f"unsupported method: {method}")

    if response.status_code != 200:
        raise RuntimeError(
            f"{method} {url} failed with HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON")
    return data


def flush_cache(base_url: str, timeout: float) -> None:
    response = request_json(
        "POST",
        base_url + "/flush_cache",
        timeout=timeout,
    )
    if response.get("status") not in {"ok", "success"}:
        raise RuntimeError(f"unexpected flush_cache response: {response}")


def load_prompt_records(dataset_path: Path) -> List[Dict[str, Any]]:
    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {dataset_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at {dataset_path}:{line_number} is not an object")
            records.append(record)
    return records


def prompt_len_for_record(record: Dict[str, Any]) -> Optional[int]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("target_prompt_tokens")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_records(
    all_records: List[Dict[str, Any]],
    prompt_len: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    selected = [
        record for record in all_records if prompt_len_for_record(record) == prompt_len
    ]
    if len(selected) < batch_size:
        raise RuntimeError(
            f"need {batch_size} records for prompt_len={prompt_len}, "
            f"found {len(selected)}"
        )
    return selected[:batch_size]


def make_chat_payload(
    record: Dict[str, Any],
    model: Optional[str],
    max_completion_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"metadata", "max_tokens", "max_completion_tokens"}
    }
    if model is not None:
        payload["model"] = model
    elif "model" not in payload:
        raise RuntimeError("record has no model; pass --model")

    payload["max_tokens"] = max_completion_tokens
    payload["temperature"] = temperature
    payload["stream"] = False
    return payload


def record_metadata(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def extract_generate_input(record: Dict[str, Any]) -> Tuple[str, Any]:
    input_ids = record.get("input_ids")
    if isinstance(input_ids, list):
        return "input_ids", input_ids

    text = record.get("text")
    if isinstance(text, str):
        return "text", text

    prompt = record.get("prompt")
    if isinstance(prompt, str):
        return "text", prompt

    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) == 1:
        message = messages[0]
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return "text", message["content"]

    raise RuntimeError(
        "generate modes require each record to contain text, prompt, input_ids, "
        "or exactly one string-content message"
    )


def make_generate_payload(
    record: Dict[str, Any],
    max_completion_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    input_field, input_value = extract_generate_input(record)
    return {
        input_field: input_value,
        "sampling_params": {
            "max_new_tokens": max_completion_tokens,
            "temperature": temperature,
            "ignore_eos": True,
        },
        "stream": False,
    }


def post_chat_request(
    base_url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]],
    timeout: float,
    barrier: threading.Barrier,
    request_index: int,
    record_metadata: Optional[Dict[str, Any]],
    save_output: bool,
) -> Dict[str, Any]:
    barrier.wait()
    start = time.perf_counter()
    data = request_json(
        "POST",
        base_url + "/v1/completions",
        payload=payload,
        headers=headers,
        timeout=timeout,
    )
    assert data["usage"]["prompt_tokens"] == record_metadata["target_prompt_tokens"], data["usage"]
    assert data["usage"]["completion_tokens"] == 128, data["usage"]
    result = {
        "ok": True,
        "request_index": request_index,
        "record_metadata": record_metadata or {},
        "client_latency_s": time.perf_counter() - start,
        "response_id": data.get("id"),
    }
    if save_output:
        result.update(chat_response_output_fields(data))
    return result


def post_generate_request(
    base_url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]],
    timeout: float,
    barrier: threading.Barrier,
    request_index: int,
    record_metadata: Optional[Dict[str, Any]],
    save_output: bool,
) -> Dict[str, Any]:
    barrier.wait()
    start = time.perf_counter()
    data = request_json(
        "POST",
        base_url + "/generate",
        payload=payload,
        headers=headers,
        timeout=timeout,
    )
    meta_info = data.get("meta_info") if isinstance(data.get("meta_info"), dict) else {}
    result = {
        "ok": True,
        "request_index": request_index,
        "record_metadata": record_metadata or {},
        "client_latency_s": time.perf_counter() - start,
        "response_id": meta_info.get("id"),
    }
    if save_output:
        result.update(generate_response_output_fields(data))
    return result


def chat_response_output_fields(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    output_text = (
        message.get("content")
        if isinstance(message, dict) and isinstance(message.get("content"), str)
        else choice.get("text")
    )
    reasoning_content = (
        message.get("reasoning_content")
        if isinstance(message, dict)
        and isinstance(message.get("reasoning_content"), str)
        else None
    )
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    return {
        "message": message if isinstance(message, dict) else {},
        "output_text": output_text,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "usage": response.get("usage"),
        "response_json": response,
    }


def generate_response_output_fields(response: Dict[str, Any]) -> Dict[str, Any]:
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    output_text = response.get("text") if isinstance(response.get("text"), str) else None
    usage = {
        "prompt_tokens": meta_info.get("prompt_tokens"),
        "completion_tokens": meta_info.get("completion_tokens"),
    }
    return {
        "output_text": output_text,
        "finish_reason": meta_info.get("finish_reason"),
        "usage": usage,
        "meta_info": meta_info,
        "response_json": response,
    }


def run_batch_requests(
    base_url: str,
    records: List[Dict[str, Any]],
    request_mode: str,
    model: Optional[str],
    max_completion_tokens: int,
    temperature: float,
    headers: Optional[Dict[str, str]],
    timeout: float,
    save_output: bool,
) -> List[Dict[str, Any]]:
    barrier = threading.Barrier(len(records))
    if request_mode in CHAT_MODES:
        payloads = [
            make_chat_payload(record, model, max_completion_tokens, temperature)
            for record in records
        ]
        post_request = post_chat_request
    elif request_mode in GENERATE_MODES:
        payloads = [
            make_generate_payload(record, max_completion_tokens, temperature)
            for record in records
        ]
        post_request = post_generate_request
    else:
        raise RuntimeError(f"unsupported request mode: {request_mode}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(records)) as pool:
        futures = [
            pool.submit(
                post_request,
                base_url,
                payload,
                headers,
                timeout,
                barrier,
                i,
                record_metadata(records[i]),
                save_output,
            )
            for i, payload in enumerate(payloads)
        ]
        return [future.result() for future in futures]


def summarize(values: Iterable[Optional[float]]) -> Tuple[str, str, str]:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean:
        return "", "", ""
    return (
        format_float(min(clean)),
        format_float(max(clean)),
        format_float(sum(clean) / len(clean)),
    )


def mean_value(values: Iterable[Optional[float]]) -> str:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean:
        return ""
    return format_float(sum(clean) / len(clean))


def percentile_value(values: Iterable[Optional[float]], percentile: float) -> str:
    clean = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not clean:
        return ""
    if len(clean) == 1:
        return format_float(clean[0])

    rank = (len(clean) - 1) * percentile / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(clean) - 1)
    fraction = rank - lo
    return format_float(clean[lo] * (1.0 - fraction) + clean[hi] * fraction)


def format_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.6f}"


def collect_request_timing_means(profile: Dict[str, Any]) -> Dict[str, str]:
    timings = profile.get("requests")
    if not isinstance(timings, list):
        return {
            "tokenization_mean_ms": "",
            "detokenization_mean_ms": "",
        }

    tokenization_ms = []
    detokenization_ms = []
    for timing in timings:
        if not isinstance(timing, dict):
            continue

        values = timing.get("timing") or {}
        tokenization = values.get("tokenization_ms")
        detokenization = values.get("detokenize_ms")
        if isinstance(tokenization, (int, float)):
            tokenization_ms.append(float(tokenization))
        if isinstance(detokenization, (int, float)):
            detokenization_ms.append(float(detokenization))

    return {
        "tokenization_mean_ms": mean_value(tokenization_ms),
        "detokenization_mean_ms": mean_value(detokenization_ms),
    }


def is_pure_full_batch_decode_step(step: Dict[str, Any], batch_size: int) -> bool:
    prefills = step.get("prefills") or []
    decoded = (step.get("decode") or {}).get("reqs") or []
    return not prefills and len(decoded) == batch_size


def full_batch_decode_window(
    steps: List[Dict[str, Any]], batch_size: int
) -> List[Dict[str, Any]]:
    window = []
    previous_step_id = None
    for step in steps:
        if not is_pure_full_batch_decode_step(step, batch_size):
            if window:
                break
            continue

        first_step_id = step.get("step_id_first", step.get("step_id"))
        if (
            window
            and previous_step_id is not None
            and first_step_id is not None
            and int(first_step_id) != previous_step_id + 1
        ):
            break

        window.append(step)
        last_step_id = step.get("step_id_last", step.get("step_id", first_step_id))
        previous_step_id = int(last_step_id) if last_step_id is not None else None
    return window


def full_batch_decode_totals(
    steps: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if not steps:
        return None, None, None

    token_counts = [(step.get("decode") or {}).get("tokens") for step in steps]
    if not all(isinstance(value, int) for value in token_counts):
        return None, None, None

    first = steps[0]
    last = steps[-1]
    start_ns = first.get("mono_ns_first")
    end_ns = last.get("mono_ns_last")
    if isinstance(start_ns, int) and isinstance(end_ns, int):
        duration_ms = (end_ns - start_ns) / 1_000_000.0
    else:
        start_time = first.get("ts_first")
        end_time = last.get("ts_last")
        if not isinstance(start_time, (int, float)) or not isinstance(
            end_time, (int, float)
        ):
            return None, None, None
        duration_ms = (float(end_time) - float(start_time)) * 1_000.0

    if duration_ms <= 0.0:
        return None, None, None
    tokens = sum(token_counts)
    return tokens, duration_ms, tokens / (duration_ms / 1_000.0)


def row_from_profile(
    prompt_len: int,
    batch_size: int,
    max_completion_tokens: int,
    profile: Dict[str, Any],
    trace_path: Path,
    outputs_path: Optional[Path] = None,
    batch_wall_ms: Optional[float] = None,
) -> Dict[str, Any]:
    steps = profile.get("steps")
    if not isinstance(steps, list):
        steps = []

    prefill_durations = []
    prefill_sampling_durations = []
    finished_requests = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        finished_requests += len(step.get("finished") or [])
        details = step.get("prefills")
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict):
                continue
            prefill_durations.append(detail.get("ms"))
            prefill_sampling_durations.append(detail.get("sampling_ms"))

    decode_steps = full_batch_decode_window(
        [step for step in steps if isinstance(step, dict)], batch_size
    )
    decode_forward_durations = []
    decode_sampling_durations = []

    for step in decode_steps:
        decode = step.get("decode") or {}
        decode_forward_durations.extend(decode.get("forward_ms_samples") or [])
        decode_sampling_durations.extend(decode.get("sampling_ms_samples") or [])

    decode_tokens, decode_duration_ms, decode_throughput = (
        full_batch_decode_totals(decode_steps)
    )

    prefill_min, prefill_max, prefill_mean = summarize(prefill_durations)
    decode_forward_min, decode_forward_max, decode_forward_mean = summarize(
        decode_forward_durations
    )
    decode_forward_p50 = percentile_value(decode_forward_durations, 50)
    decode_forward_p90 = percentile_value(decode_forward_durations, 90)
    request_timing_means = collect_request_timing_means(profile)

    row = {
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "max_completion_tokens": max_completion_tokens,
        "start_step_id": profile.get("start_step_id", ""),
        "end_step_id": profile.get("end_step_id", ""),
        "trace_record_count": len(steps),
        "finished_requests": finished_requests,
        "batch_wall_ms": (
            format_float(batch_wall_ms)
            if isinstance(batch_wall_ms, (int, float))
            else ""
        ),
        "prefill_count": len([value for value in prefill_durations if value is not None]),
        "prefill_min_ms": prefill_min,
        "prefill_max_ms": prefill_max,
        "prefill_mean_ms": prefill_mean,
        "prefill_sampling_mean_ms": mean_value(prefill_sampling_durations),
        "decode_full_batch_steps": len(decode_forward_durations),
        "decode_full_batch_tokens": (
            decode_tokens if decode_tokens is not None else ""
        ),
        "decode_full_batch_duration_ms": (
            format_float(decode_duration_ms)
            if decode_duration_ms is not None
            else ""
        ),
        "decode_throughput": (
            format_float(decode_throughput)
            if decode_throughput is not None
            else ""
        ),
        "decode_forward_min_ms": decode_forward_min,
        "decode_forward_max_ms": decode_forward_max,
        "decode_forward_mean_ms": decode_forward_mean,
        "decode_forward_p50_ms": decode_forward_p50,
        "decode_forward_p90_ms": decode_forward_p90,
        "decode_sampling_mean_ms": mean_value(decode_sampling_durations),
        "sampling_mean_ms": mean_value(
            list(prefill_sampling_durations) + list(decode_sampling_durations)
        ),
        "tokenization_mean_ms": request_timing_means["tokenization_mean_ms"],
        "detokenization_mean_ms": request_timing_means["detokenization_mean_ms"],
        "trace_json": str(trace_path),
        "outputs_jsonl": str(outputs_path) if outputs_path is not None else "",
    }
    return row


def write_csv_row(csv_path: Path, row: Dict[str, Any], reset: bool):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = reset or not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open(newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != CSV_FIELDS:
            raise ValueError(
                f"CSV header does not match the current schema: {csv_path}. "
                "Use --reset or choose a new --output-csv path."
            )
    mode = "w" if reset and write_header else "a"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def trace_window_path(output_dir: Path, prompt_len: int, batch_size: int) -> Path:
    return output_dir / f"trace_len{prompt_len}_bs{batch_size}.json"


def raw_outputs_path(output_dir: Path, prompt_len: int, batch_size: int) -> Path:
    return output_dir / f"outputs_len{prompt_len}_bs{batch_size}.jsonl"


def write_outputs_jsonl(
    output_path: Path,
    request_results: List[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in sorted(
            request_results,
            key=lambda item: int(item.get("request_index", 0)),
        ):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_cell(
    base_url: str,
    records: List[Dict[str, Any]],
    request_mode: str,
    prompt_len: int,
    batch_size: int,
    model: Optional[str],
    max_completion_tokens: int,
    temperature: float,
    trace_dir: Path,
    timeout: float,
    save_outputs: bool,
    flush_cache_between_cells: bool,
) -> Dict[str, Any]:
    selected = select_records(records, prompt_len, batch_size)
    headers = (
        {SKIP_PREFILL_HEADER: "true"}
        if request_mode in SKIP_PREFILL_MODES
        else None
    )

    print(f"\n>>> mode={request_mode} prompt_len={prompt_len} batch_size={batch_size}")
    if flush_cache_between_cells:
        print("    flushing KV cache before profile cell")
        flush_cache(base_url, timeout=timeout)

    try:
        trace_index = request_json(
            "GET", base_url + "/debug/trace", timeout=timeout
        )
        if not trace_index.get("step_samples", False):
            raise RuntimeError("server must be launched with --trace-step-samples")
        before = request_json(
            "GET", base_url + "/debug/scheduler_metrics", timeout=timeout
        )
        current = before.get("current") or {}
        if current.get("waiting") or current.get("running"):
            raise RuntimeError("scheduler must be idle before a trace benchmark")
        start_step_id = int((before.get("totals") or {}).get("total_steps", 0))
        start_wall = time.time()
        batch_start = time.perf_counter()
        request_results = run_batch_requests(
            base_url=base_url,
            records=selected,
            request_mode=request_mode,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            headers=headers,
            timeout=timeout,
            save_output=save_outputs,
        )
        batch_wall_ms = (time.perf_counter() - batch_start) * 1000.0
        end_wall = time.time()
        after = request_json(
            "GET", base_url + "/debug/scheduler_metrics", timeout=timeout
        )
        end_step_id = int((after.get("totals") or {}).get("total_steps", 0))
        steps = collect_trace_pages(
            lambda params: request_json(
                "GET",
                base_url + "/debug/trace/steps",
                payload=params,
                timeout=timeout,
            ),
            "records",
            params={
                "after_step_id": start_step_id,
                "through_step_id": end_step_id,
                "order": "asc",
            },
        )
        requests_in_window = collect_trace_pages(
            lambda params: request_json(
                "GET",
                base_url + "/debug/trace/requests",
                payload=params,
                timeout=timeout,
            ),
            "requests",
            params={"since": start_wall, "until": end_wall, "order": "asc"},
        )
        profile = {
            "start_step_id": start_step_id,
            "end_step_id": end_step_id,
            "steps": steps,
            "requests": requests_in_window,
        }
    finally:
        if flush_cache_between_cells:
            print("    flushing KV cache after profile cell")
            flush_cache(base_url, timeout=timeout)

    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_window_path(trace_dir, prompt_len, batch_size)
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
        f.write("\n")

    outputs_path = None
    if save_outputs:
        outputs_path = raw_outputs_path(trace_dir, prompt_len, batch_size)
        write_outputs_jsonl(outputs_path, request_results)

    return row_from_profile(
        prompt_len=prompt_len,
        batch_size=batch_size,
        max_completion_tokens=max_completion_tokens,
        profile=profile,
        trace_path=trace_path,
        outputs_path=outputs_path,
        batch_wall_ms=batch_wall_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument(
        "--mode",
        choices=[
            MODE_CHAT,
            MODE_CHAT_SKIP_PREFILL,
            MODE_GENERATE,
            MODE_GENERATE_SKIP_PREFILL,
        ],
        default=MODE_CHAT,
        help=(
            "Request mode: chat uses normal /v1/chat/completions; "
            "chat-skip-prefill uses chat with X-Score-Skip-Prefill; "
            "generate uses native /generate; generate-skip-prefill uses "
            "/generate with X-Score-Skip-Prefill"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmark_data/chat_prompts_war_and_peace.jsonl"),
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark_data/profile_sweep.csv"))
    parser.add_argument("--trace-dir", type=Path, default=Path("benchmark_data/profile_sweep_traces"))
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help=(
            "Save one JSONL record per request with response JSON, output text, "
            "usage/meta_info, finish reason, and request metadata"
        ),
    )
    parser.add_argument(
        "--flush-cache-between-cells",
        action="store_true",
        help=(
            "Flush /flush_cache before and after each profiled "
            "prompt-length/batch-size cell. Requests inside a cell still run "
            "concurrently."
        ),
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Idle this long between prompt-length/batch-size cells so the host cools down",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")
    if any(length <= 0 for length in args.prompt_lengths):
        raise ValueError("--prompt-lengths must contain positive integers")
    if any(size <= 0 for size in args.batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if args.max_completion_tokens <= 0:
        raise ValueError("--max-completion-tokens must be positive")
    if args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must be non-negative")

    records = load_prompt_records(args.dataset)
    base_url = normalize_base_url(args.base_url)

    first_write = True
    rows = []
    for prompt_len in args.prompt_lengths:
        for batch_size in args.batch_sizes:
            if args.cooldown_seconds and not first_write:
                print(f"  cooling down {args.cooldown_seconds:g}s", flush=True)
                time.sleep(args.cooldown_seconds)
            row = run_cell(
                base_url=base_url,
                records=records,
                request_mode=args.mode,
                prompt_len=prompt_len,
                batch_size=batch_size,
                model=args.model,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                trace_dir=args.trace_dir,
                timeout=args.timeout,
                save_outputs=args.save_outputs,
                flush_cache_between_cells=args.flush_cache_between_cells,
            )
            write_csv_row(args.output_csv, row, reset=args.reset and first_write)
            first_write = False
            rows.append(row)
            print(
                "    "
                f"prefill_mean={row['prefill_mean_ms'] or '-'}ms "
                f"decode_forward_mean={row['decode_forward_mean_ms'] or '-'}ms "
                f"decode_forward_p50={row['decode_forward_p50_ms'] or '-'}ms "
                f"decode_forward_p90={row['decode_forward_p90_ms'] or '-'}ms "
                f"decode_full_batch_steps={row['decode_full_batch_steps']} "
                "decode_throughput="
                f"{row['decode_throughput'] or '-'}tok/s"
            )

    print(f"\nwrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
