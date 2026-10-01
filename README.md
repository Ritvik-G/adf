# Architectural Decision Framework

A benchmark comparing seven ways of invoking the same deterministic action, evaluating an arithmetic expression, as a stand-in for the architectural choices a system makes when deciding how to execute a function or tool call. The methods fall into four families:

| Family | Method | How it works |
|---|---|---|
| Function calling | **Function Calling** | Python's `eval()`, in-process. The baseline. |
| Service-oriented | **REST** | The expression is split into pairwise operations, each sent as an HTTP POST to a Flask endpoint (`/add`, `/subtract`, ...). |
| | **SOAP** | The same split, each operation sent as an XML SOAP envelope to a single endpoint that also serves a WSDL contract. |
| | **MCP** | The same split, each operation sent as a Model Context Protocol tool call to a server exposing four tools. No LLM involved. |
| LLM, direct | **LLM-API** (Groq) | A hosted LLM is asked for the numerical answer directly. |
| | **Ollama LLM** | Locally hosted LLMs are asked the same way. |
| LLM + tools | **LLM + MCP Tools** | A local LLM is given the four MCP tools and must solve the equation by calling them, then reply with the number. |

Every method is scored on the same 1,000 equations for **accuracy**, **time**, **cost** and **development effort** (lines of code). See [Methodology](#methodology).

## Repository layout

```
code/              all scripts (see "What each script does")
data/              the 1,000-equation dataset (SVAMP train + test)
results/           the paper's results: latest.json, one run_*.json per configuration,
                   and figures/ (paper figures + Table 1)
requirements.txt   pinned dependencies (tested with Python 3.13.12)
.env.example       template for the only secret, a Groq API key (optional)
```

## Reproducing the paper's figures and table

The paper's results are committed in `results/latest.json`. To rebuild Figures 1-3 and Table 1 from them:

```bash
pip install -r requirements.txt
cd code
python paper_figures.py
```

This writes to `results/figures/`: vector PDFs (for LaTeX) and 600 dpi PNGs of each figure, plus `table_results.tex`. No servers or models are needed.

## Running the benchmark yourself

### 1. Setup

```bash
python3 -m venv agentic_compositions
source agentic_compositions/bin/activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and pull the models used in the paper:

```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull phi4-mini:3.8b
ollama pull granite4:3b
ollama pull qwen2-math:1.5b
ollama pull qwen3.5:4b
```

For the Groq methods (optional), add an API key:

```bash
cp .env.example .env    # then set GROQ_API_KEY=your_key
```

`code/config_llm.json` holds all non-secret settings: the Groq model and URL, the models for each sweep (`ollama_models` for direct runs, `ollama_agent_models` for agents), the `temperatures` to sweep, `max_tokens`, and `agent_max_steps`.

### 2. Start the servers

Each in its own terminal, from `code/` with the venv active. Leave them running.

```bash
python rest_server.py    # http://localhost:5001
python soap_server.py    # http://localhost:8000
python mcp_server.py     # http://localhost:8100/mcp
```

REST uses port 5001 because on macOS the AirPlay Receiver occupies port 5000 and answers every request with 403. Ollama must also be running (`ollama serve`, or the Ollama app).

### 3. Run

From `code/`:

```bash
# Deterministic methods (seconds)
for m in vanilla rest soap mcp; do python benchmark.py --method $m; done

# Direct LLMs: every model in ollama_models at every temperature (about 1.5 h)
caffeinate -i python benchmark.py --method ollama

# LLM + MCP tools: every model in ollama_agent_models at every temperature (about 8 h)
caffeinate -i python benchmark.py --method mcp_agent_ollama

# Groq, direct and as an agent (needs GROQ_API_KEY)
python benchmark.py --method groq
python benchmark.py --method mcp_agent_groq
```

Useful options: `-n 20` runs only the first 20 equations, and `--ollama-model <name>` runs a single model. `caffeinate -i` (macOS) keeps the machine awake during long runs.

A method whose server, API key or model is unavailable is skipped with a message. Each configuration is saved as soon as it finishes, to `results/run_<timestamp>_<method>.json`, and merged into `results/latest.json`, so an interrupted run keeps everything completed so far.

`latest.json` already contains the paper's results, and a new run replaces the entry for the same configuration. To start from an empty file, move `results/latest.json` aside first.

For a quick check of one method without the full measurement harness:

```bash
python main.py --method mcp -n 20
python main.py --method mcp_agent_ollama --ollama-model granite4:3b -n 20
```

### 4. (Optional) CPU power sampling

`benchmark.py` can also record CPU power through macOS `powermetrics`, which needs passwordless sudo. Add this line with `sudo visudo`, replacing `yourusername`:

```
yourusername ALL=(root) NOPASSWD: /usr/sbin/powermetrics
```

Without it, the power fields are `null`. Power is not part of the paper's comparison.

## What each script does

The pipeline is: dataset → `benchmark.py` (with the servers running) → `results/*.json` → `paper_figures.py` → `results/figures/`.

| Script | Role |
|---|---|
| `config.py`, `config_llm.json` | Paths, dataset loading and settings. The Groq key is read from `.env`. |
| `monitor.py` | CPU-time and memory measurement, used by the benchmark and by each server's `/metrics` endpoint. |
| `rest_server.py`, `rest_client.py` | REST service and the client that splits the expression into pairwise operations, one POST each. |
| `soap_server.py`, `soap_client.py` | The same with SOAP envelopes, a WSDL contract and SOAP faults. |
| `mcp_server.py`, `mcp_client.py` | Four MCP tools, and a client that runs the same split over one persistent MCP session. |
| `mcp_agent.py` | The agent loop: turns the MCP tools into LLM tool definitions, calls Groq or Ollama, and runs the tools the model asks for. |
| `main.py` | One function per method, the strict scoring rule (`classify_result`), and a quick single-method CLI. |
| `benchmark.py` | The measurement harness. Runs methods and sweeps, records every metric, and saves the results. |
| `paper_figures.py` | Builds the paper's figures and Table 1 from `latest.json`. |
| `plot_results.py` | Quick-look charts of whatever is in `latest.json`. Readable for a handful of methods, cluttered for all 26 configurations. |
| `final_plots.py` | Legacy script behind the original version of the paper's figures. It reads result files that are no longer in the repo and no longer runs. |

## Methodology

**Strict scoring.** A response is correct only if it parses directly as the correct number. There is no retry and no extraction of a number from longer text. Every equation falls into exactly one outcome: `correct`, `wrong_answer` (a clean number with the wrong value), `format_failure` (anything that is not a number alone) or `api_error` (a failed or rejected request, or an agent that exceeded the turn limit). See `code/main.py::classify_result`.

**Temperature.** Local models are run at both `T=0` (greedy decoding) and `T=0.7` (a common conversational default). Temperature is sent under Ollama's `options`, the only place Ollama reads it. An earlier version sent it at the top level, where Ollama silently ignores it. Results carry the temperature in their name (`ollama_llm:<model>:T<temp>`) and in a `temperature` field. The cloud API runs at `T=0.7`.

**LLM + MCP Tools.** The model receives the MCP server's four tools, converted from its tool listing, and a system prompt telling it to use them and reply with only the final number. Each tool call runs on the MCP server and its result, or its error, is returned to the model. The loop ends when the model replies without a tool call, which is graded as its final answer, or after `agent_max_steps` (6) LLM turns, which is scored as `api_error`. Each reply is limited to `max_tokens` (256) so a model stuck repeating itself stops early. Direct Ollama calls have no length limit and a 30 s timeout per request. Agent runs also record `llm_turns`, `tool_calls`, `tool_errors`, `no_tool_answers` and `max_steps_exceeded`. `qwen2-math:1.5b` does not support tool calling, so all of its agent requests are rejected. `qwen3.5:4b` is only run as an agent, because answering directly it frequently exceeded the 30 s timeout.

**Cost** is measured in the unit that fits each method:
- Function Calling, REST, SOAP and MCP: client CPU time, as one before-and-after delta of `psutil`'s process counters. Servers also report their own CPU time through `/metrics`.
- Local LLMs: Ollama's self-reported inference time (prompt processing plus generation), which covers CPU and GPU execution. For agents it is summed over every LLM turn.
- Cloud API: token consumption.

## Environment and data

The paper's results were produced on an Apple M1 Pro with 16 GB of RAM, running Ollama 0.34.4 and Python 3.13.12. Absolute times will differ on other hardware.

- `data/train.json`, `data/test.json`: the 1,000 SVAMP equations (`Equation`, `Answer`), combined for every run. Each equation has at most two operations, 1.236 on average.
- The cloud API row in the paper (gpt-oss-20b, 15.5% accuracy, 306.35 s) comes from an earlier run of the Groq method. Its result file is not in the repo, so `paper_figures.py` sets those two values in the `API_BASELINE` constant.

## License and citation

The code is released under the [MIT License](LICENSE). If you use it or its results, please cite the paper; GitHub's "Cite this repository" button (from [`CITATION.cff`](CITATION.cff)) provides the reference.
