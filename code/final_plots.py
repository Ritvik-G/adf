"""
Final comparison charts for the write-up. Reads a fixed, hand-curated set of
result files (not the running results/latest.json) and produces:

  - final_accuracy_overview.png   — accuracy, every config (standalone)
  - final_temperature_effect.png  — temp 0 vs 0.7, per ollama model
  - final_outcome_breakdown.png   — correct / wrong / format_failure / api_error, ollama models
  - final_measures.png            — Accuracy / Total Time / CPU-seconds, 3-panel centerpiece figure
  - final_efficiency_scatter.png  — accuracy vs time, every config

Color: one fixed hue per model/method identity (never per chart), reused
across every figure. Temperature is a variant of an identity, not a new
identity, so it's encoded as solid vs hatched fill on the same hue —
never a separate hue.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, "..", "results")
DATA_DIR = os.path.join(CODE_DIR, "..", "data")

# ============================================================================
# Palette (validated: node scripts/validate_palette.js — all checks pass)
# ============================================================================
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
STATUS_GOOD = "#0ca30c"

IDENTITY_COLOR = {
    "fc": "#2a78d6",
    "rest": "#eb6834",
    "soap": "#1baf7a",
    "groq": "#eda100",
    "llama3.2": "#e87ba4",
    "phi4": "#008300",
    "qwen2": "#4a3aa7",
}
FAILURE_COLOR = {
    "wrong_answer": "#2a78d6",
    "format_failure": "#eb6834",
    "api_error": "#1baf7a",
}

# ============================================================================
# Font sizes — sized for print legibility, not just on-screen viewing.
# ============================================================================
FONT_SUPTITLE = 26
FONT_TITLE = 20
FONT_AXIS_LABEL = 16
FONT_TICK = 13
FONT_VALUE = 12
FONT_LEGEND = 13
FONT_ANNOTATION = 12

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# ============================================================================
# File set — the curated final selection, not the running latest.json
# ============================================================================
FILES = [
    {"path": os.path.join(RESULTS_DIR, "vanilla.json"), "label": "Function\nCalling", "identity": "fc", "temp": None},
    {"path": os.path.join(RESULTS_DIR, "rest.json"), "label": "rest", "identity": "rest", "temp": None},
    {"path": os.path.join(RESULTS_DIR, "soap.json"), "label": "soap", "identity": "soap", "temp": None},
    {"path": os.path.join(DATA_DIR, "original_groq_baseline.json"), "label": "API (gpt-oss-20b)", "identity": "groq", "temp": None},
    {"path": os.path.join(RESULTS_DIR, "llama3.2_0_updated.json"), "label": "llama3.2:1b\n(T=0)", "identity": "llama3.2", "temp": 0.0},
    {"path": os.path.join(RESULTS_DIR, "llama3.2_7_updated.json"), "label": "llama3.2:1b\n(T=0.7)", "identity": "llama3.2", "temp": 0.7},
    {"path": os.path.join(RESULTS_DIR, "phi4_0_updated.json"), "label": "phi4-mini:3.8b\n(T=0)", "identity": "phi4", "temp": 0.0},
    {"path": os.path.join(RESULTS_DIR, "phi4_7_updated.json"), "label": "phi4-mini:3.8b\n(T=0.7)", "identity": "phi4", "temp": 0.7},
    {"path": os.path.join(RESULTS_DIR, "qwen2_0_updated.json"), "label": "qwen2-math:1.5b\n(T=0)", "identity": "qwen2", "temp": 0.0},
    {"path": os.path.join(RESULTS_DIR, "qwen2_7_updated.json"), "label": "qwen2-math:1.5b\n(T=0.7)", "identity": "qwen2", "temp": 0.7},
]


def load_records():
    """Load and normalize every file in FILES into a common schema, so the
    old (pre-refactor) groq result and the new nested schema can be charted
    side by side.

    processor_runtime_s is the "Cost" metric shown in the Measures chart:
      - fc/rest/soap: client cpu_seconds — accurate as-is, no GPU involved.
      - groq: None — a remote API's cost is token consumption, not local compute.
      - ollama models: Ollama's own self-reported inference time
        (prompt_eval_duration + eval_duration), which reflects both CPU and
        GPU execution on this system — not the client's cpu_seconds, which
        only sees the cost of sending the request and waiting for a reply.
    """
    records = []
    for meta in FILES:
        with open(meta["path"], "r", encoding="utf-8") as f:
            raw = json.load(f)
        failures = raw.get("failures")

        ollama_timing = raw.get("ollama_timing")
        if ollama_timing is not None:
            processor_runtime_s = (
                ollama_timing["eval_duration_ns"] + ollama_timing["prompt_eval_duration_ns"]
            ) / 1e9
        else:
            processor_runtime_s = raw.get("cpu_seconds")  # None for the old groq schema

        records.append({
            "label": meta["label"],
            "identity": meta["identity"],
            "temp": meta["temp"],
            "n": raw.get("n"),
            "accuracy": raw["accuracy"],
            "total_time_s": raw["total_time_s"],
            "cpu_seconds": raw.get("cpu_seconds"),  # None for the old groq schema
            "processor_runtime_s": processor_runtime_s,
            "avg_memory_mb": raw.get("avg_memory_mb"),
            "failures": failures,  # None for the old groq schema (no breakdown available)
        })
    return records


def bar_style(identity, temp):
    """Solid fill for temp 0 / no-temp methods; hatched, lighter fill for
    temp 0.7 — temperature is a variant of the identity's color, not a new
    hue."""
    color = IDENTITY_COLOR[identity]
    if temp == 0.7:
        return {"facecolor": color, "edgecolor": color, "alpha": 0.45, "hatch": "///"}
    return {"facecolor": color, "edgecolor": color, "alpha": 1.0, "hatch": None}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ============================================================================
# Chart 1 — Accuracy overview
# ============================================================================

def draw_accuracy_bars(ax, records, value_fontsize=FONT_VALUE):
    """Shared bar-drawing logic — used by the standalone accuracy chart and
    by the Accuracy panel inside the Measures composite, so both stay
    visually identical without duplicating the drawing code."""
    for i, r in enumerate(records):
        style = bar_style(r["identity"], r["temp"])
        ax.bar(i, r["accuracy"], width=0.6, zorder=3, **style)
        ax.text(i, r["accuracy"] + 1.5, f"{r['accuracy']:.1f}%", ha="center", va="bottom",
                fontsize=value_fontsize, color=INK)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_AXIS_LABEL)


def plot_accuracy_overview(records, output_path):
    fig, ax = plt.subplots(figsize=(17, 7.5))
    x = np.arange(len(records))

    draw_accuracy_bars(ax, records)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in records], fontsize=FONT_TICK, rotation=25, ha="right", color=INK_SECONDARY)
    ax.set_title("Accuracy by Method / Model  (n=1000)", fontsize=FONT_TITLE, fontweight="bold", color=INK, pad=16)
    style_axes(ax)

    solid_patch = mpatches.Patch(facecolor=MUTED, edgecolor=MUTED, label="deterministic / T=0")
    hatch_patch = mpatches.Patch(facecolor=MUTED, edgecolor=MUTED, alpha=0.45, hatch="///", label="T=0.7")
    ax.legend(handles=[solid_patch, hatch_patch], loc="upper right", frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved → {output_path}")


# ============================================================================
# Chart 2 — Temperature effect
# ============================================================================

def plot_temperature_effect(records, output_path):
    models = ["llama3.2", "phi4", "qwen2"]
    display_names = {"llama3.2": "llama3.2:1b", "phi4": "phi4-mini:3.8b", "qwen2": "qwen2-math:1.5b"}
    by_model = {m: {r["temp"]: r for r in records if r["identity"] == m} for m in models}

    fig, ax = plt.subplots(figsize=(10, 7))
    width = 0.32
    x = np.arange(len(models))

    for i, m in enumerate(models):
        r0, r7 = by_model[m][0.0], by_model[m][0.7]
        s0, s7 = bar_style(m, 0.0), bar_style(m, 0.7)
        ax.bar(i - width / 2, r0["accuracy"], width=width, zorder=3, **s0)
        ax.bar(i + width / 2, r7["accuracy"], width=width, zorder=3, **s7)
        ax.text(i - width / 2, r0["accuracy"] + 1.5, f"{r0['accuracy']:.1f}%", ha="center", fontsize=FONT_VALUE, color=INK)
        ax.text(i + width / 2, r7["accuracy"] + 1.5, f"{r7['accuracy']:.1f}%", ha="center", fontsize=FONT_VALUE, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([display_names[m] for m in models], fontsize=FONT_TICK, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Temperature Effect on Accuracy  (T=0 vs T=0.7)", fontsize=FONT_TITLE, fontweight="bold", color=INK, pad=16)
    style_axes(ax)

    solid_patch = mpatches.Patch(facecolor=MUTED, edgecolor=MUTED, label="T=0")
    hatch_patch = mpatches.Patch(facecolor=MUTED, edgecolor=MUTED, alpha=0.45, hatch="///", label="T=0.7")
    ax.legend(handles=[solid_patch, hatch_patch], loc="upper right", frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved → {output_path}")


# ============================================================================
# Chart 3 — Outcome breakdown (ollama models only — the old groq run has no
# failure-category breakdown to compare against)
# ============================================================================

def plot_outcome_breakdown(records, output_path):
    ollama_records = [r for r in records if r["identity"] in ("llama3.2", "phi4", "qwen2")]

    fig, ax = plt.subplots(figsize=(14, 7.5))
    x = np.arange(len(ollama_records))

    for i, r in enumerate(ollama_records):
        n = r["n"]
        f = r["failures"]
        correct = n - sum(f.values())
        bottom = 0
        for count, color, label in [
            (correct, STATUS_GOOD, "correct"),
            (f["wrong_answer"], FAILURE_COLOR["wrong_answer"], "wrong_answer"),
            (f["format_failure"], FAILURE_COLOR["format_failure"], "format_failure"),
            (f["api_error"], FAILURE_COLOR["api_error"], "api_error"),
        ]:
            ax.bar(i, count, bottom=bottom, width=0.6, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
            bottom += count

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in ollama_records], fontsize=FONT_TICK, rotation=20, ha="right", color=INK_SECONDARY)
    ax.set_ylim(0, 1050)
    ax.set_ylabel("Count (of 1000 samples)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Outcome Breakdown — Ollama Models", fontsize=FONT_TITLE, fontweight="bold", color=INK, pad=16)
    style_axes(ax)

    handles = [
        mpatches.Patch(color=STATUS_GOOD, label="correct"),
        mpatches.Patch(color=FAILURE_COLOR["wrong_answer"], label="wrong_answer"),
        mpatches.Patch(color=FAILURE_COLOR["format_failure"], label="format_failure"),
        mpatches.Patch(color=FAILURE_COLOR["api_error"], label="api_error"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.30, 1.0), frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved → {output_path}")


# ============================================================================
# Chart 4 — Measures (3-panel centerpiece: Accuracy / Total Time / Processor Runtime)
#
# Memory (avg RSS) was dropped: it measures the benchmark client process's
# own footprint (Python + numpy + matplotlib already loaded), not the LLM's
# actual memory cost — the model runs inside a separate Ollama process this
# pipeline never queries. Showing it invited a false read (e.g. "vanilla
# used more memory than an LLM"), so it's removed rather than relabeled.
#
# Processor Runtime (panel C) is client cpu_seconds for fc/rest/soap (accurate
# as-is — pure CPU workloads), Ollama's own self-reported inference time for
# the local models (covers both CPU and GPU execution on this system, unlike
# client cpu_seconds which only sees request/response overhead), and n/a for
# the remote API (its cost metric is token consumption, shown separately).
# ============================================================================

def plot_measures(records, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    x = np.arange(len(records))

    # --- Panel A: Accuracy ---
    ax = axes[0]
    draw_accuracy_bars(ax, records, value_fontsize=FONT_VALUE - 1)
    ax.set_title("Accuracy", fontsize=FONT_TITLE - 2, fontweight="bold", color=INK)

    # --- Panel B: Total time (log) ---
    ax = axes[1]
    for i, r in enumerate(records):
        style = bar_style(r["identity"], r["temp"])
        ax.bar(i, r["total_time_s"], width=0.6, zorder=3, **style)
    ax.set_yscale("log")
    ax.set_ylabel("Total time (s)  [log scale]", fontsize=FONT_AXIS_LABEL - 2)
    ax.set_title("Total Time", fontsize=FONT_TITLE - 2, fontweight="bold", color=INK)

    # --- Panel C: Processor Runtime (log) ---
    ax = axes[2]
    for i, r in enumerate(records):
        style = bar_style(r["identity"], r["temp"])
        val = r["processor_runtime_s"]
        if val is None:
            ax.text(i, 0.01, "n/a\n(token\ncost)", ha="center", va="bottom", fontsize=FONT_VALUE - 2, color=MUTED)
            continue
        ax.bar(i, val, width=0.6, zorder=3, **style)
    ax.set_yscale("log")
    ax.set_ylabel("Processor runtime (s)  [log scale]", fontsize=FONT_AXIS_LABEL - 2)
    ax.set_title("Processor Runtime", fontsize=FONT_TITLE - 2, fontweight="bold", color=INK)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in records], fontsize=FONT_TICK - 1, rotation=30, ha="right", color=INK_SECONDARY)
        style_axes(ax)

    fig.suptitle("Measures", fontsize=FONT_SUPTITLE, fontweight="bold", color=INK, y=1.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


# ============================================================================
# Chart 5 — Efficiency scatter (accuracy vs time)
# ============================================================================


# Several configs land on nearly the same point (temp 0 vs 0.7 barely moves
# accuracy/time; rest and soap are near-identical). Default same-corner
# offsets collide there, so each point gets an explicit offset + a thin
# leader line back to its marker.
SCATTER_OFFSETS = {
    ("fc", None): (10, 6),
    ("rest", None): (14, 16),
    ("soap", None): (14, -18),
    ("groq", None): (10, 6),
    ("llama3.2", 0.0): (16, 18),
    ("llama3.2", 0.7): (16, -20),
    ("phi4", 0.0): (16, 18),
    ("phi4", 0.7): (16, -20),
    ("qwen2", 0.0): (18, 14),
    ("qwen2", 0.7): (18, -20),
}


GOODNESS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("goodness", [SURFACE, "#b7d3f6"])


def draw_goodness_gradient(ax, xlim, ylim, n=300):
    """
    Background wash indicating which region of the plot is better, replacing
    a directional arrow drawn through the data. Higher accuracy (right) and
    lower time (down, on a log axis) is better, so goodness increases toward
    the bottom-right. Built in log-y space (np.logspace + log10 normalization)
    so the gradient still reads as smooth and diagonal after the log-scale
    axis transform — a plain linear-space gradient would band badly across
    six orders of magnitude.
    """
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), n)
    X, Y = np.meshgrid(xs, ys)
    norm_x = (X - xlim[0]) / (xlim[1] - xlim[0])
    norm_logy = (np.log10(Y) - np.log10(ylim[0])) / (np.log10(ylim[1]) - np.log10(ylim[0]))
    goodness = 0.5 * norm_x + 0.5 * (1 - norm_logy)
    ax.pcolormesh(X, Y, goodness, cmap=GOODNESS_CMAP, vmin=0, vmax=1,
                  shading="gouraud", zorder=0, alpha=0.6)


def draw_arrow_axes(ax, xlim, ylim):
    """Open, arrow-tipped x/y axes instead of a boxed spine rectangle."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(1, 0), xytext=(0, 0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AXIS, linewidth=1.8, mutation_scale=18),
                annotation_clip=False)
    ax.annotate("", xy=(0, 1), xytext=(0, 0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=AXIS, linewidth=1.8, mutation_scale=18),
                annotation_clip=False)


def plot_efficiency_scatter(records, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    xlim = (-5, 110)
    ylim = (0.003, 6000)

    draw_goodness_gradient(ax, xlim, ylim)

    for r in records:
        color = IDENTITY_COLOR[r["identity"]]
        marker = "^" if r["temp"] == 0.7 else "o"
        fill = "none" if r["temp"] == 0.7 else color
        ax.scatter(r["accuracy"], r["total_time_s"], s=170, marker=marker,
                   facecolors=fill, edgecolors=color, linewidths=2.2, zorder=5)
        dx, dy = SCATTER_OFFSETS[(r["identity"], r["temp"])]
        ax.annotate(
            r["label"].replace("\n", " "), xy=(r["accuracy"], r["total_time_s"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=FONT_ANNOTATION, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=0, shrinkB=4),
        )

    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Accuracy (%)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Total time (s)  [log scale]", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Accuracy vs Time\n(bottom-right = ideal)", fontsize=FONT_TITLE, fontweight="bold", color=INK, pad=16)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    draw_arrow_axes(ax, xlim, ylim)

    circle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=10, label="T=0 / deterministic")
    triangle = plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="none", markeredgecolor=MUTED, markersize=10, label="T=0.7")
    ax.legend(handles=[circle, triangle], loc="lower left", frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved → {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = load_records()
    print(f"Loaded {len(records)} result files\n")

    plot_accuracy_overview(records, os.path.join(RESULTS_DIR, "final_accuracy_overview.png"))
    plot_temperature_effect(records, os.path.join(RESULTS_DIR, "final_temperature_effect.png"))
    plot_outcome_breakdown(records, os.path.join(RESULTS_DIR, "final_outcome_breakdown.png"))
    plot_measures(records, os.path.join(RESULTS_DIR, "final_measures.png"))
    plot_efficiency_scatter(records, os.path.join(RESULTS_DIR, "final_efficiency_scatter.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
