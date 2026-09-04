# Agentic Decision Framework

A benchmark comparing five distinct ways of invoking the same deterministic action — evaluating a mathematical expression — as a stand-in for the architectural choices an agentic system makes when deciding how to execute a tool call:

- **Function Calling** — direct in-process evaluation, no network, no serialization.
- **REST** — the same computation decomposed into HTTP calls against a Flask service, one call per arithmetic operation.
- **SOAP** — the same decomposition, but over XML-enveloped SOAP requests against a single endpoint.
- **Remote LLM API** (Groq) — a hosted LLM asked to compute the answer directly from natural-language instruction.
- **Local LLM** (Ollama) — one or more locally-hosted models, asked the same way, at two temperature settings each.

Every method is scored on the same 1,000-equation dataset along three axes — **Accuracy**, **Time**, and **Cost** — with each axis measured in the unit that's actually meaningful for that method, rather than forced into one universal number (see [Methodology](#methodology)).

## Repository layout

```
code/               all scripts
data/                the equation dataset + a historical baseline result
results/             benchmark output (JSON) and generated charts (gitignored except via explicit copy)
requirements.txt
.env.example         template for the one secret this project needs (a Groq API key)
```

## 1. Setup

```bash
python3 -m venv agentic_compositions
source agentic_compositions/bin/activate
pip install -r requirements.txt
```

Add your Groq key:

```bash
cp .env.example .env
# edit .env, set GROQ_API_KEY=your_real_key
```

`code/config_llm.json` holds everything non-secret — the Groq model name/URL, the list of Ollama models to sweep (`ollama_models`), temperature, and token limits. No secrets live in it; the API key is injected from `.env` at runtime (`code/config.py`).

Pull whichever Ollama models you intend to run, e.g.:

```bash
ollama pull llama3.2:1b
ollama pull phi4-mini:3.8b
ollama pull qwen2-math:1.5b
```

## 2. (Optional) enable CPU wattage sampling

Wattage uses macOS `powermetrics`, which needs root. To let it run without a password prompt each time:

```bash
sudo visudo
```

Add this line (replace `yourusername`):

```
yourusername ALL=(root) NOPASSWD: /usr/sbin/powermetrics
```

If you skip this, the benchmark still runs — the power fields just report `null`. This is a secondary, best-effort metric; it is not part of the core Accuracy/Time/Cost comparison.

## 3. Start the REST and SOAP servers

Each in its own terminal (both need the venv active):

```bash
source agentic_compositions/bin/activate
cd code
python rest_server.py       # http://localhost:5000
```

```bash
source agentic_compositions/bin/activate
cd code
python soap_server.py       # http://localhost:8000
```

Leave both running. (Ollama is separate — `ollama serve`, if you're testing that method.)

## 4. Run a single method (quick check)

```bash
source agentic_compositions/bin/activate
cd code
python main.py --method vanilla                          # Function Calling
python main.py --method soap -n 20                        # limit to 20 samples
python main.py --method ollama --ollama-model phi4-mini:3.8b
```

## 5. Run the full benchmark

```bash
cd code
python benchmark.py --method all      # or vanilla / rest / soap / groq / ollama
python benchmark.py --method ollama   # sweeps every model in ollama_models
python benchmark.py --method ollama --ollama-model qwen2-math:1.5b   # one model only
```

Each invocation skips any method whose server/API isn't reachable and says so. Results are written per-method (`run_<timestamp>_<method>.json`) and merged into `results/latest.json`, so you can build up a full comparison one `--method` at a time rather than needing every server/model available in a single run. `benchmark.py` only measures and saves — it never plots (see below).

## 6. Generate charts

Two separate scripts, two separate purposes:

- **`python plot_results.py`** — reads whatever is currently in `results/latest.json` and produces a general-purpose comparison set: accuracy/time bars, an accuracy-vs-time scatter, raw-unit resource bars, and a fixed-baseline spider chart. This reflects whatever you've actually run.
- **`python final_plots.py`** — reads a fixed, hand-curated set of specific result files (the ones behind this project's own writeup) and produces the five figures used there: an accuracy overview, a temperature-effect comparison, an Ollama outcome breakdown, a 3-panel "Measures" figure (Accuracy / Time / Processor Runtime), and an accuracy-vs-time efficiency scatter. Edit the `FILES` list at the top of that script to point at your own result files if you want to reproduce this specific figure set with new data.

## Methodology

**Strict, no-leniency scoring.** Every method's raw output must parse directly as a number to count as correct — there is no retry, no regex-based answer extraction, and no salvage of a number embedded in prose. For the LLM methods, this is deliberate: the prompt asks for a number-only reply, so a verbose response (even one that computes the right answer) is scored as a **format failure**, not a correct answer. Every sample is classified into exactly one of `correct` / `wrong_answer` / `format_failure` / `api_error` (`code/main.py::classify_result`), so accuracy numbers can be decomposed into *why* a method failed, not just whether it did.

**Cost is measured per-method, not forced into one unit**, because "cost" means structurally different things depending on where the computation happens:
- Function Calling / REST / SOAP are pure CPU workloads with no separate process — client-side CPU-seconds is a direct, accurate cost measure.
- The remote API's real cost is metered token consumption, not local CPU time (the computation happens on someone else's infrastructure).
- Local Ollama models report their own inference duration (prompt evaluation + generation time, in `code/benchmark.py::llm_ollama_instrumented`) — this covers both CPU and GPU execution on the host machine, and is used instead of client-side CPU-seconds, which would only capture the trivial cost of sending a request and waiting for a reply.

**Temperature.** Every Ollama model is run at both `T=0` (greedy decoding, deterministic given fixed weights) and `T=0.7` (the common default for conversational use), so any claim about temperature affecting accuracy is backed by a direct comparison rather than assumption.

## Data

- `data/train.json`, `data/test.json` — the 1,000-equation dataset (`Equation`, `Answer` pairs), combined and used in full by every benchmark run.
- `data/original_groq_baseline.json` — a fixed historical result from an earlier version of the Groq evaluation path, kept as a reference baseline in `final_plots.py`; it predates the current `ollama_timing`/failure-classification instrumentation and has no comparable breakdown data.
