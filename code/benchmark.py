"""
Comparative Benchmark for Mathematical Expression Evaluators
Runs methods and saves metrics to JSON. No plotting here — run
paper_figures.py (paper figures and table) or plot_results.py (quick look)
separately once you have the results you want charted.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
import datetime
from functools import partial
import requests

from main import vanilla, rest_calls, soap_arch, extract_number, classify_result
from config import load_dataset, load_llm_config
from monitor import SystemMonitor
from mcp_client import connect
from main import mcp_arch
from mcp_agent import run_agent, tools_for_llm, chat_groq, groq_tool_message, chat_ollama, ollama_tool_message

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MCP_SERVER_URL = "http://localhost:8100"


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

def llm_ollama_instrumented(equation=None, model=None, temperature=None):
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

    if temperature is None:
        temperature = cfg.get("temperature", 0.7)

    prompt = f"Answer with ONLY the final numerical value, no explanations or equations. QUESTION: {equation}\nANSWER:"

    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Ollama only reads sampling settings from "options"; a top-level
        # "temperature" is silently ignored.
        "options": {"temperature": temperature},
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

def run_benchmark(method_name, func, n, dataset, is_groq=False, is_ollama=False, is_agent=False, server_url=None):
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
        is_agent (bool): If True (used with is_groq or is_ollama), the returned
            dict also carries MCP agent counts (llm_turns, tool_calls, ...),
            which are aggregated too.
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
    total_agent = {"llm_turns": 0, "tool_calls": 0, "tool_errors": 0, "no_tool_answers": 0, "max_steps_exceeded": 0}

    for i in range(min(n, len(dataset))):
        item = dataset[i]
        equation = item.get("Equation", "")
        expected = item.get("Answer", "")
        stats = {}

        try:
            if is_groq:
                result, stats = func(equation=equation)
                for k in total_tokens:
                    total_tokens[k] += stats.get(k, 0)
            elif is_ollama:
                result, stats = func(equation=equation)
                for k in total_ollama_timing:
                    total_ollama_timing[k] += stats.get(k, 0)
            else:
                result = func(equation=equation)
        except Exception as e:
            print(f"  ❌  API_ERROR {equation!r}: {e}")
            result = None

        if is_agent:
            for k in total_agent:
                total_agent[k] += stats.get(k, 0)

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

    if is_agent:
        result["agent"] = total_agent
        result["avg_tool_calls"] = round(total_agent["tool_calls"] / n, 3) if n > 0 else 0
        result["avg_llm_turns"] = round(total_agent["llm_turns"] / n, 3) if n > 0 else 0

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
    if is_agent:
        print(f"  ✓  Agent: {total_agent}  |  avg tool calls: {result['avg_tool_calls']}  |  avg LLM turns: {result['avg_llm_turns']}")
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
    """
    Return True if the URL answers 200 within timeout seconds. A 200 is
    required, not just any reply: another program on the same port (e.g.
    macOS's AirPlay Receiver on 5000) would otherwise pass as our server.
    """
    try:
        return requests.get(url, timeout=timeout).status_code == 200
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
    if not check_server("http://localhost:5001/metrics"):
        print("\n[SKIP] REST server not reachable at localhost:5001 — start rest_server.py first")
        return None
    return run_benchmark("rest", rest_calls, n, data, server_url="http://localhost:5001")


def run_soap(n, data, args):
    if not check_server("http://localhost:8000/metrics"):
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
    Runs every model in config_llm.json's "ollama_models" list (unless
    --ollama-model overrides it to a single model), once per temperature in
    "temperatures". Each is its own result, method name
    "ollama_llm:<model>:T<temp>".
    A generator: yields each result as soon as it finishes, so main()
    saves it straight away.
    """
    if not check_server("http://localhost:11434"):
        print("\n[SKIP] Ollama not reachable at localhost:11434 — start Ollama first")
        return

    cfg = load_llm_config()
    models = [args.ollama_model] if getattr(args, "ollama_model", None) else cfg.get("ollama_models", ["mistral"])

    for model in models:
        if not ollama_has_model(model):
            print(f"\n[SKIP] Ollama model {model} not pulled — run `ollama pull {model}` first")
            continue
        for temperature in cfg["temperatures"]:
            result = run_benchmark(
                f"ollama_llm:{model}:T{temperature}",
                lambda equation=None, m=model, t=temperature: llm_ollama_instrumented(
                    equation=equation, model=m, temperature=t
                ),
                n,
                data,
                is_ollama=True,
            )
            result["temperature"] = temperature
            yield result


def run_mcp(n, data, args):
    if not check_server(f"{MCP_SERVER_URL}/metrics"):
        print("\n[SKIP] MCP server not reachable at localhost:8100 — start mcp_server.py first")
        return None
    with connect() as session:
        return run_benchmark(
            "mcp",
            lambda equation=None: mcp_arch(equation=equation, session=session),
            n,
            data,
            server_url=MCP_SERVER_URL,
        )


def run_mcp_agent_groq(n, data, args):
    """
    Runs the Groq agent once per temperature in config_llm.json's
    "temperatures" (method name "mcp_agent_groq:T<temp>").
    A generator: yields each result as soon as it finishes, so main()
    saves it straight away.
    """
    if not check_server(f"{MCP_SERVER_URL}/metrics"):
        print("\n[SKIP] MCP server not reachable at localhost:8100 — start mcp_server.py first")
        return
    cfg = load_llm_config()
    if not cfg.get("api_key") or not cfg.get("model"):
        print("\n[SKIP] MCP agent (Groq): missing GROQ_API_KEY (.env) or model in config_llm.json")
        return

    with connect() as session:
        tools = tools_for_llm(session)
        for temperature in cfg["temperatures"]:
            chat = partial(chat_groq, temperature=temperature)
            result = run_benchmark(
                f"mcp_agent_groq:T{temperature}",
                lambda equation=None, chat=chat: run_agent(
                    equation, session, chat, groq_tool_message, tools, cfg["agent_max_steps"]
                ),
                n,
                data,
                is_groq=True,
                is_agent=True,
                server_url=MCP_SERVER_URL,
            )
            result["temperature"] = temperature
            yield result


def ollama_has_model(model):
    """Return True if this model is pulled in the local Ollama."""
    r = requests.get("http://localhost:11434/api/tags", timeout=2)
    names = [m["name"] for m in r.json().get("models", [])]
    return model in names or f"{model}:latest" in names


def run_mcp_agent_ollama(n, data, args):
    """
    Runs every model in config_llm.json's "ollama_agent_models" list (unless
    --ollama-model overrides it to a single model), once per temperature in
    "temperatures". Each is its own result, method name
    "mcp_agent_ollama:<model>:T<temp>".
    A generator: yields each result as soon as it finishes, so main()
    saves it straight away.
    """
    if not check_server(f"{MCP_SERVER_URL}/metrics"):
        print("\n[SKIP] MCP server not reachable at localhost:8100 — start mcp_server.py first")
        return
    if not check_server("http://localhost:11434"):
        print("\n[SKIP] Ollama not reachable at localhost:11434 — start Ollama first")
        return

    cfg = load_llm_config()
    models = [args.ollama_model] if getattr(args, "ollama_model", None) else cfg["ollama_agent_models"]

    with connect() as session:
        tools = tools_for_llm(session)
        for model in models:
            if not ollama_has_model(model):
                print(f"\n[SKIP] Ollama model {model} not pulled — run `ollama pull {model}` first")
                continue
            for temperature in cfg["temperatures"]:
                chat = partial(chat_ollama, model=model, temperature=temperature)
                result = run_benchmark(
                    f"mcp_agent_ollama:{model}:T{temperature}",
                    lambda equation=None, chat=chat: run_agent(
                        equation, session, chat, ollama_tool_message, tools, cfg["agent_max_steps"]
                    ),
                    n,
                    data,
                    is_ollama=True,
                    is_agent=True,
                    server_url=MCP_SERVER_URL,
                )
                result["temperature"] = temperature
                yield result


METHOD_RUNNERS = {
    "vanilla": run_vanilla,
    "rest": run_rest,
    "soap": run_soap,
    "groq": run_groq,
    "ollama": run_ollama,
    "mcp": run_mcp,
    "mcp_agent_groq": run_mcp_agent_groq,
    "mcp_agent_ollama": run_mcp_agent_ollama,
}


# ============================================================================
# Main
# ============================================================================

def print_summary_table(results):
    print(f"\n{'='*142}")
    print(f"{'Method':<34} {'Accuracy':>10} {'Time(s)':>10} {'CPU%':>8} {'CPU-s':>8} {'Mem(MB)':>10} {'Tokens/tok-per-s':>18}  {'Failures (wrong/format/api_error)':>34}")
    print(f"{'-'*142}")
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
            f"{r['method']:<34}"
            f" {r['accuracy']:>9.1f}%"
            f" {r['total_time_s']:>10.2f}"
            f" {r['avg_cpu_percent']:>7.1f}%"
            f" {r['cpu_seconds']:>8.2f}"
            f" {r['avg_memory_mb']:>9.1f}"
            f"  {tok:>18}"
            f"  {fail_str:>34}"
        )
    print(f"{'='*142}\n")


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

    # Merge into the running combined file — lets you build up vanilla/rest/soap/
    # ollama/groq one `--method` at a time instead of needing them all in one pass.
    combined = load_latest()
    for method in methods_to_run:
        r = METHOD_RUNNERS[method](n, data, args)
        if r is None:
            continue
        # A runner returns one result, or (Ollama and MCP agents) a generator
        # yielding results one by one. Each is saved as soon as it arrives,
        # so a long run that gets interrupted keeps everything finished so far.
        for res in ([r] if isinstance(r, dict) else r):
            safe_name = res["method"].replace(":", "-")
            save_result(res, os.path.join(RESULTS_DIR, f"run_{timestamp}_{safe_name}.json"))
            combined = upsert_result(combined, res)
            combined["timestamp"] = timestamp
            combined["n"] = n
            save_result(combined, os.path.join(RESULTS_DIR, "latest.json"))

    save_result(combined, os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json"))
    print(f"\n  Latest results → {os.path.join(RESULTS_DIR, 'latest.json')}  ({len(combined['results'])} methods so far)")
    print("  Run `python paper_figures.py` to (re)generate the paper figures and table from latest.json.")

    print_summary_table(combined["results"])


if __name__ == "__main__":
    main()
