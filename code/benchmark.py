"""
Comparative Benchmark for Mathematical Expression Evaluators
Runs methods and saves metrics to JSON. No plotting here — run
plot_results.py separately once you have the results you want charted.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
import datetime
import requests

from main import vanilla, rest_calls, soap_arch, extract_number, classify_result
from config import load_dataset, load_llm_config
from monitor import SystemMonitor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# ============================================================================
# CPU wattage via macOS powermetrics (optional — needs sudo)
# ============================================================================

class WattMonitor:
    """
    Samples CPU package power via macOS `powermetrics` in a background thread.
    Requires passwordless sudo for powermetrics (see README setup notes).
    If it can't run (no sudo, not macOS, tool missing), stats() returns None
    values and the rest of the benchmark proceeds unaffected.
    """

    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self._samples = []
        self._proc = None
        self._thread = None
        self._running = False

    def start(self):
        self._samples = []
        self._running = True
        try:
            self._proc = subprocess.Popen(
                ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-i", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            self._proc = None
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        pattern = re.compile(r"CPU Power:\s*([\d.]+)\s*mW")
        for line in self._proc.stdout:
            if not self._running:
                break
            match = pattern.search(line)
            if match:
                self._samples.append(float(match.group(1)))

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2)

    def stats(self):
        if not self._samples:
            return {"avg_cpu_power_mw": None, "peak_cpu_power_mw": None}
        return {
            "avg_cpu_power_mw": round(sum(self._samples) / len(self._samples), 2),
            "peak_cpu_power_mw": round(max(self._samples), 2),
        }


# ============================================================================
# Groq with token capture
# ============================================================================

def llms_groq_instrumented(equation=None):
    """
    Like llms_groq in main.py, but also returns token usage from the API
    response. Single attempt, no retry: an API error is scored as an API
    error, not retried away.
    Returns (raw_result_str, usage_dict).
    """
    if equation is None:
        raise ValueError("Equation not provided")

    cfg = load_llm_config()
    api_key = cfg.get("api_key")
    model = cfg.get("model")
    url = cfg.get("url")

    if not api_key or not model:
        raise ValueError("Config must include 'api_key' and 'model'")

    prompt = f"Answer with ONLY the final numerical value, no explanations or equations. QUESTION: {equation}\nANSWER:"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.7),
        "max_tokens": cfg.get("max_tokens", 256),
        "top_p": cfg.get("top_p", 1.0),
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        raw = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return raw, usage
    except Exception as e:
        print(f"  Groq API Error: {e}")
        return None, {}


# ============================================================================
# Ollama with server-side timing capture
#
# Ollama's /api/generate response includes its own timing breakdown
# (nanoseconds), measured inside Ollama's own process — this is the actual
# inference cost, unlike the client's CPU-seconds/wall-clock time, which
# only sees the cost of sending the request and waiting for a reply.
# ============================================================================

def llm_ollama_instrumented(equation=None, model=None):
    """
    Like llm_ollama in main.py, but also returns Ollama's own server-side
    timing fields from the API response.
    Returns (raw_result_str, timing_dict). timing_dict is {} on failure.
    """
    if equation is None:
        raise ValueError("Equation not provided")

    cfg = load_llm_config()
    model = model or cfg.get("ollama_models", ["mistral"])[0]
    base_url = cfg.get("ollama_url", "http://localhost:11434")

    prompt = f"Answer with ONLY the final numerical value, no explanations or equations. QUESTION: {equation}\nANSWER:"

    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": cfg.get("temperature", 0.7),
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        res = data.get("response", "").strip()
        timing = {
            "total_duration_ns": data.get("total_duration", 0),
            "load_duration_ns": data.get("load_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "eval_duration_ns": data.get("eval_duration", 0),
        }
        return res, timing
    except requests.exceptions.ConnectionError:
        print(f"  Ollama Error: Could not connect to {base_url}")
        return None, {}
    except Exception as e:
        print(f"  Ollama Error: {e}")
        return None, {}


# ============================================================================
# Server-side metrics (REST / SOAP)
# ============================================================================

def reset_server_metrics(base_url):
    try:
        requests.post(f"{base_url}/metrics/reset", timeout=2)
    except Exception:
        pass


def fetch_server_metrics(base_url):
    try:
        r = requests.get(f"{base_url}/metrics", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ============================================================================
# Core benchmark runner
# ============================================================================

def run_benchmark(method_name, func, n, dataset, is_groq=False, is_ollama=False, server_url=None):
    """
    Run a single evaluation method over n samples and collect all metrics.

    Args:
        method_name (str): Display name for this method.
        func: Callable with signature func(equation=...) -> result, or
              (equation=...) -> (result, usage) when is_groq=True, or
              (equation=...) -> (result, timing) when is_ollama=True.
        n (int): Number of samples.
        dataset (list): List of {"Equation": ..., "Answer": ...} dicts.
        is_groq (bool): If True, func returns (result, usage_dict) and tokens are tracked.
        is_ollama (bool): If True, func returns (result, timing_dict) and Ollama's
            own server-side timing fields (real inference cost, not the
            client's proxy for it) are aggregated.
        server_url (str): If set, fetch this server's own /metrics before and
            after the run, to report server-side CPU/memory separately from
            client-side CPU/memory.

    Returns:
        dict: All collected metrics for this method.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {method_name}")
    print(f"{'='*60}")

    if server_url:
        reset_server_metrics(server_url)

    client_monitor = SystemMonitor()
    watt_monitor = WattMonitor()
    client_monitor.start()
    watt_monitor.start()
    start_time = time.perf_counter()

    counts = {"correct": 0, "wrong_answer": 0, "format_failure": 0, "api_error": 0}
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_ollama_timing = {
        "total_duration_ns": 0,
        "load_duration_ns": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration_ns": 0,
        "eval_count": 0,
        "eval_duration_ns": 0,
    }

    for i in range(min(n, len(dataset))):
        item = dataset[i]
        equation = item.get("Equation", "")
        expected = item.get("Answer", "")

        try:
            if is_groq:
                result, usage = func(equation=equation)
                for k in total_tokens:
                    total_tokens[k] += usage.get(k, 0)
            elif is_ollama:
                result, timing = func(equation=equation)
                for k in total_ollama_timing:
                    total_ollama_timing[k] += timing.get(k, 0)
            else:
                result = func(equation=equation)
        except Exception as e:
            print(f"  ❌  API_ERROR {equation!r}: {e}")
            result = None

        category = classify_result(result, expected)
        counts[category] += 1

        if category == "wrong_answer":
            print(f"  ❌  WRONG_ANSWER {equation} → Got: {result}, Expected: {expected}")
        elif category == "format_failure":
            hint = extract_number(str(result))
            note = f" (contained {hint}, not counted — instruction was number-only)" if hint else ""
            print(f"  ❌  FORMAT_FAILURE {equation} → raw: {result!r}{note}")
        elif category == "api_error" and result is not None:
            print(f"  ❌  API_ERROR {equation} → no result returned")

    end_time = time.perf_counter()
    client_monitor.stop()
    watt_monitor.stop()

    total_time = end_time - start_time
    accuracy = (counts["correct"] / n) * 100 if n > 0 else 0.0
    client_stats = client_monitor.stats()
    watt_stats = watt_monitor.stats()

    result = {
        "method": method_name,
        "n": n,
        "correct": counts["correct"],
        "errors": n - counts["correct"],
        "failures": {k: v for k, v in counts.items() if k != "correct"},
        "accuracy": round(accuracy, 2),
        "total_time_s": round(total_time, 4),
        "avg_time_per_sample_s": round(total_time / n, 6) if n > 0 else 0,
        "client": client_stats,
        "power": watt_stats,
    }
    # Flatten the client-side keys used elsewhere (spider, printing, old scripts)
    result.update(client_stats)

    if server_url:
        server_stats = fetch_server_metrics(server_url)
        result["server"] = server_stats

    if is_groq:
        result["tokens"] = total_tokens

    if is_ollama:
        result["ollama_timing"] = total_ollama_timing
        eval_s = total_ollama_timing["eval_duration_ns"] / 1e9
        result["ollama_tokens_per_sec"] = round(total_ollama_timing["eval_count"] / eval_s, 2) if eval_s > 0 else None

    print(
        f"  ✓  Accuracy: {accuracy:.1f}%  |  Time: {total_time:.2f}s"
        f"  |  Client CPU: {client_stats['avg_cpu_percent']}%"
        f"  |  Client CPU-seconds: {client_stats['cpu_seconds']}s"
        f"  |  Client Mem: {client_stats['avg_memory_mb']:.1f} MB"
    )
    print(f"  ✓  Failures: {result['failures']}")
    if is_ollama:
        tps = result["ollama_tokens_per_sec"]
        print(
            f"  ✓  Ollama server-side: {total_ollama_timing['eval_count']} tokens generated"
            f"  |  {tps if tps is not None else 'n/a'} tok/s"
            f"  |  total_duration: {total_ollama_timing['total_duration_ns'] / 1e9:.2f}s"
        )
    if server_url:
        print(f"  ✓  Server-side: {result.get('server')}")
    if watt_stats["avg_cpu_power_mw"] is not None:
        print(f"  ✓  CPU Power: {watt_stats['avg_cpu_power_mw']} mW avg")
    else:
        print("  ⚠  CPU Power: unavailable (needs passwordless sudo for powermetrics)")

    return result


# ============================================================================
# Helpers
# ============================================================================

def check_server(url, timeout=2):
    """Return True if the URL responds within timeout seconds."""
    try:
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False


def save_result(result, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_latest():
    """Load the running combined results file, or start a fresh one."""
    path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"timestamp": None, "n": None, "results": []}


def upsert_result(combined, result):
    """Replace this method's entry in the combined results (if present) with the new one."""
    combined["results"] = [r for r in combined["results"] if r["method"] != result["method"]]
    combined["results"].append(result)
    return combined


# ============================================================================
# Per-method runners — each returns a result dict, or None (and prints why
# it was skipped) if that method's server/API isn't available right now.
# ============================================================================

def run_vanilla(n, data, args):
    return run_benchmark("vanilla", vanilla, n, data)


def run_rest(n, data, args):
    if not check_server("http://localhost:5000"):
        print("\n[SKIP] REST server not reachable at localhost:5000 — start rest_server.py first")
        return None
    return run_benchmark("rest", rest_calls, n, data, server_url="http://localhost:5000")


def run_soap(n, data, args):
    if not check_server("http://localhost:8000"):
        print("\n[SKIP] SOAP server not reachable at localhost:8000 — start soap_server.py first")
        return None
    return run_benchmark("soap", soap_arch, n, data, server_url="http://localhost:8000")


def run_groq(n, data, args):
    try:
        cfg = load_llm_config()
    except Exception as e:
        print(f"\n[SKIP] Groq error: {e}")
        return None
    if not cfg.get("api_key") or not cfg.get("model"):
        print("\n[SKIP] Groq: missing GROQ_API_KEY (.env) or model in config_llm.json")
        return None
    return run_benchmark("groq_llm", llms_groq_instrumented, n, data, is_groq=True)


def run_ollama(n, data, args):
    """
    Runs every model in config_llm.json's "ollama_models" list, each as its
    own result (method name "ollama_llm:<model>") — unless --ollama-model
    overrides it to a single model. Returns a list of result dicts (or None
    if Ollama itself isn't reachable).
    """
    if not check_server("http://localhost:11434"):
        print("\n[SKIP] Ollama not reachable at localhost:11434 — start Ollama first")
        return None

    cfg = load_llm_config()
    models = [args.ollama_model] if getattr(args, "ollama_model", None) else cfg.get("ollama_models", ["mistral"])

    results = []
    for model in models:
        results.append(
            run_benchmark(
                f"ollama_llm:{model}",
                lambda equation=None, m=model: llm_ollama_instrumented(equation=equation, model=m),
                n,
                data,
                is_ollama=True,
            )
        )
    return results


METHOD_RUNNERS = {
    "vanilla": run_vanilla,
    "rest": run_rest,
    "soap": run_soap,
    "groq": run_groq,
    "ollama": run_ollama,
}


# ============================================================================
# Main
# ============================================================================

def print_summary_table(results):
    print(f"\n{'='*128}")
    print(f"{'Method':<20} {'Accuracy':>10} {'Time(s)':>10} {'CPU%':>8} {'CPU-s':>8} {'Mem(MB)':>10} {'Tokens/tok-per-s':>18}  {'Failures (wrong/format/api_error)':>34}")
    print(f"{'-'*128}")
    for r in results:
        if "tokens" in r:
            tok = str(r["tokens"].get("total_tokens", "-"))
        elif "ollama_tokens_per_sec" in r:
            tps = r["ollama_tokens_per_sec"]
            tok = f"{tps} tok/s" if tps is not None else "n/a"
        else:
            tok = "-"
        f = r["failures"]
        fail_str = f"{f['wrong_answer']}/{f['format_failure']}/{f['api_error']}"
        print(
            f"{r['method']:<20}"
            f" {r['accuracy']:>9.1f}%"
            f" {r['total_time_s']:>10.2f}"
            f" {r['avg_cpu_percent']:>7.1f}%"
            f" {r['cpu_seconds']:>8.2f}"
            f" {r['avg_memory_mb']:>9.1f}"
            f"  {tok:>18}"
            f"  {fail_str:>34}"
        )
    print(f"{'='*128}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the comparative benchmark, with full metrics")
    parser.add_argument(
        "--method",
        choices=["all"] + sorted(METHOD_RUNNERS.keys()),
        default="all",
        help="Which method to run (default: all)",
    )
    parser.add_argument("-n", type=int, default=None, help="Number of samples (default: full dataset)")
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Run only this ollama model, overriding the ollama_models list in config_llm.json",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    train, test = load_dataset()
    data = train + test
    n = args.n if args.n is not None else len(data)
    print(f"\nDataset: {len(train)} train + {len(test)} test = {len(data)} total samples  |  running {n}")

    methods_to_run = list(METHOD_RUNNERS) if args.method == "all" else [args.method]

    new_results = []
    for method in methods_to_run:
        r = METHOD_RUNNERS[method](n, data, args)
        if r is None:
            continue
        for res in (r if isinstance(r, list) else [r]):
            new_results.append(res)
            safe_name = res["method"].replace(":", "-")
            save_result(res, os.path.join(RESULTS_DIR, f"run_{timestamp}_{safe_name}.json"))

    # Merge into the running combined file — lets you build up vanilla/rest/soap/
    # ollama/groq one `--method` at a time instead of needing them all in one pass.
    combined = load_latest()
    for r in new_results:
        combined = upsert_result(combined, r)
    combined["timestamp"] = timestamp
    combined["n"] = n
    save_result(combined, os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json"))
    save_result(combined, os.path.join(RESULTS_DIR, "latest.json"))
    print(f"\n  Latest results → {os.path.join(RESULTS_DIR, 'latest.json')}  ({len(combined['results'])} methods so far)")
    print("  Run `python plot_results.py` separately to (re)generate charts from latest.json.")

    print_summary_table(combined["results"])


if __name__ == "__main__":
    main()
