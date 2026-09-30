"""
Paper figures, drawn at their printed size and saved as vector PDFs.

Reads results/latest.json and writes to results/figures/:
  - final_measures.pdf              — Accuracy / Time / Cost, direct vs + MCP tools
  - final_efficiency_scatter.pdf    — accuracy vs time, arrows from direct to + MCP tools
  - final_outcome_breakdown.pdf     — correct / wrong / format / api_error per model and mode
  - final_temperature_effect.pdf    — T=0 vs T=0.7 accuracy per model and mode
  - final_outcome_distribution.pdf  — the two above side by side, (a) and (b)
plus a 600 dpi PNG of each, for slides and previews, and
  - table_results.tex               — the paper's results table (Table 1), as LaTeX

Why vector PDFs at printed size: a PDF holding a screenshot of a chart (what a
design-tool export produces) pixelates when zoomed or printed however many
pixels it has. Here text stays text and bars stay shapes, and every figure is
drawn at the width it's printed at (AAAI: 7.0 in text width), so fonts are
their real 6-8 pt size instead of being shrunk by LaTeX.

LLM values are the mean over T=0 and T=0.7 (as in the paper's results table),
except in the temperature figure, which shows both.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib.colors
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TEXT_WIDTH_IN = 7.0  # AAAI \textwidth

# The cloud API row is kept from the earlier run reported in the paper
# (data/original_groq_baseline.json is not in this repo). Replace these two
# values with that file's once it's available.
API_BASELINE = {"accuracy": 15.5, "total_time_s": 306.35}

# ============================================================================
# Palette — same as final_plots.py, with new identities added
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
    "mcp": "#8c564b",
    "groq": "#eda100",
    "llama3.2:1b": "#e87ba4",
    "llama3.2:3b": "#b8386e",
    "phi4": "#008300",
    "granite": "#17a2b8",
    "qwen2": "#4a3aa7",
    "qwen3.5": "#9b7fe0",
}
OUTCOME_COLOR = {
    "correct": STATUS_GOOD,
    "wrong_answer": "#2a78d6",
    "format_failure": "#eb6834",
    "api_error": "#1baf7a",
}
OUTCOME_LABEL = {
    "correct": "Correct",
    "wrong_answer": "Wrong answer",
    "format_failure": "Format failure",
    "api_error": "API err. / step limit",
}

plt.rcParams.update({
    "pdf.fonttype": 42,  # embed fonts as real text, not outlines
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": INK_SECONDARY,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# ============================================================================
# Which runs appear, in display order. Each LLM model has a direct run
# ("ollama_llm"), a with-tools run ("mcp_agent_ollama"), or both.
# qwen2-math is direct only (it does not support tool calling);
# qwen3.5 is with-tools only (answering directly it often exceeded 30 s).
# ============================================================================
DETERMINISTIC = [
    ("Function Calling", "fc", "vanilla"),
    ("REST", "rest", "rest"),
    ("SOAP", "soap", "soap"),
    ("MCP", "mcp", "mcp"),
]
LLM_MODELS = [
    # label, identity, ollama model, has direct run, has with-tools run
    ("Llama3.2-1B", "llama3.2:1b", "llama3.2:1b", True, True),
    ("Llama3.2-3B", "llama3.2:3b", "llama3.2:3b", True, True),
    ("Phi4-mini-3.8B", "phi4", "phi4-mini:3.8b", True, True),
    ("Granite4-3B", "granite", "granite4:3b", True, True),
    ("Qwen2-Math-1.5B", "qwen2", "qwen2-math:1.5b", True, False),
    ("Qwen3.5-4B", "qwen3.5", "qwen3.5:4b", False, True),
]
TEMPERATURES = (0.0, 0.7)


# ============================================================================
# Data
# ============================================================================

def load_results():
    with open(os.path.join(RESULTS_DIR, "latest.json"), "r", encoding="utf-8") as f:
        return {r["method"]: r for r in json.load(f)["results"]}


def cost_seconds(r):
    """
    The paper's Cost metric: client CPU-seconds for deterministic methods;
    Ollama's own inference time (prompt processing + generation, CPU and
    GPU) for local LLMs, since the client only waits on those.
    """
    timing = r.get("ollama_timing")
    if timing:
        return (timing["prompt_eval_duration_ns"] + timing["eval_duration_ns"]) / 1e9
    return r["cpu_seconds"]


def summarize(runs):
    """Mean accuracy / time / cost / outcome counts over a list of runs (one per temperature)."""
    k = len(runs)
    return {
        "accuracy": sum(r["accuracy"] for r in runs) / k,
        "total_time_s": sum(r["total_time_s"] for r in runs) / k,
        "cost_s": sum(cost_seconds(r) for r in runs) / k,
        "correct": sum(r["correct"] for r in runs) / k,
        "failures": {f: sum(r["failures"][f] for r in runs) / k for f in runs[0]["failures"]},
        "by_temp": {r["temperature"]: r["accuracy"] for r in runs if "temperature" in r},
    }


def llm_runs(results, prefix, model):
    return [results[f"{prefix}:{model}:T{t}"] for t in TEMPERATURES]


def build_rows(results):
    """One dict per displayed method/model: direct and/or with-tools summaries."""
    rows = []
    for label, identity, key in DETERMINISTIC:
        rows.append({"label": label, "identity": identity, "group": "Deterministic",
                     "direct": summarize([results[key]]), "tools": None})
    rows.append({"label": "API (gpt-oss-20b)$^\\dagger$", "identity": "groq", "group": "LLM",
                 "direct": {"accuracy": API_BASELINE["accuracy"], "total_time_s": API_BASELINE["total_time_s"],
                            "cost_s": None},
                 "tools": None})
    for label, identity, model, has_direct, has_tools in LLM_MODELS:
        rows.append({
            "label": label, "identity": identity, "group": "LLM",
            "direct": summarize(llm_runs(results, "ollama_llm", model)) if has_direct else None,
            "tools": summarize(llm_runs(results, "mcp_agent_ollama", model)) if has_tools else None,
        })
    return rows


# ============================================================================
# Shared styling
# ============================================================================

def bar_style(identity, mode):
    """Direct (and deterministic): solid fill. With MCP tools: lighter fill,
    hatched, same hue — a variant of the model, not a new identity."""
    color = IDENTITY_COLOR[identity]
    if mode == "tools":
        light = matplotlib.colors.to_rgba(color, 0.35)
        return {"facecolor": light, "edgecolor": color, "hatch": "//////", "linewidth": 0.6}
    return {"facecolor": color, "edgecolor": color, "linewidth": 0.6}


def mode_legend_handles():
    return [
        mpatches.Patch(facecolor=MUTED, edgecolor=MUTED, linewidth=0.6, label="Direct invocation"),
        mpatches.Patch(facecolor=matplotlib.colors.to_rgba(MUTED, 0.35), edgecolor=MUTED,
                       hatch="//////", linewidth=0.6, label="LLM + MCP tools"),
    ]


def style_axes(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=2)


def fmt_seconds(v):
    if v < 0.1:
        return f"{v:.4f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:,.0f}"


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf = os.path.join(FIGURES_DIR, f"{name}.pdf")
    png = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved → {pdf}  (+ .png)")


# ============================================================================
# Figure 1 — Measures: Accuracy / Time / Cost, as horizontal bars sharing one
# set of rows. Paired models get two bars: direct (solid) and + tools (hatched).
# ============================================================================

def layout_rows(rows):
    """
    Y positions (top to bottom) for group headers, row labels and bars.
    Returns (ticks, labels, bars) where bars = [(y, row, mode)].
    """
    y = 0.0
    ticks, labels, bars = [], [], []
    group = None
    for row in rows:
        if row["group"] != group:
            group = row["group"]
            if ticks:
                y += 0.35
            ticks.append(y)
            labels.append(group)
            y += 0.9
        modes = [m for m in ("direct", "tools") if row[m] is not None]
        paired = row["direct"] is not None and row["tools"] is not None
        ticks.append(y + (0.21 if paired else 0))
        labels.append(row["label"])
        if paired:
            bars += [(y, row, "direct"), (y + 0.42, row, "tools")]
            y += 1.25
        else:
            bars.append((y, row, modes[0]))
            y += 0.9
    return ticks, labels, bars


def plot_measures(rows):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    ticks, labels, bars = layout_rows(rows)
    height = 0.38

    panels = [
        ("(a) Accuracy (%)", "accuracy", False, (0, 118)),
        ("(b) Total time (s, log scale)", "total_time_s", True, (1e-3, 2e5)),
        ("(c) Cost (s, log scale)", "cost_s", True, (1e-3, 2e5)),
    ]
    for ax, (title, key, log, xlim) in zip(axes, panels):
        for y, row, mode in bars:
            value = row[mode][key]
            if value is None:  # the API has no local compute cost; its cost is tokens
                ax.text(xlim[0] * 1.5, y, "n/a (token-billed)", va="center", fontsize=5.5, color=MUTED)
                continue
            left = xlim[0] if log else 0
            ax.barh(y, value - left, left=left, height=height, zorder=3, **bar_style(row["identity"], mode))
            text = f"{value:.1f}" if key == "accuracy" else fmt_seconds(value)
            ax.text(value * (1.25 if log else 1) + (0 if log else 1.5), y, text,
                    va="center", fontsize=5.5, color=INK)
        if log:
            ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        style_axes(ax, grid_axis="x")

    ax = axes[0]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for tick in ax.get_yticklabels():
        if tick.get_text() in ("Deterministic", "LLM"):
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.invert_yaxis()

    fig.legend(handles=mode_legend_handles(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, -0.04))
    save(fig, "final_measures")


# ============================================================================
# Figure 2 — Efficiency scatter: accuracy vs time. Arrows join each model's
# direct point to its + MCP tools point, so the shift tools cause is visible.
# ============================================================================

GOODNESS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("goodness", [SURFACE, "#b7d3f6"])

# Only these points get a text label (offset in points); LLM models are
# identified by the colour legend instead, which avoids overlapping labels.
SCATTER_LABELS = {
    "fc": ("Function Calling", (-7, 0)),
    "rest": ("REST / SOAP / MCP", (-7, 0)),
    "groq": ("API (gpt-oss-20b)†", (6, 0)),
}


def draw_goodness_gradient(ax, xlim, ylim, n=200):
    """Background wash: better toward the bottom-right (higher accuracy, less time)."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), n)
    X, Y = np.meshgrid(xs, ys)
    norm_x = (X - xlim[0]) / (xlim[1] - xlim[0])
    norm_logy = (np.log10(Y) - np.log10(ylim[0])) / (np.log10(ylim[1]) - np.log10(ylim[0]))
    goodness = 0.5 * norm_x + 0.5 * (1 - norm_logy)
    # The only non-vector element: rasterized at the save dpi so it stays smooth
    ax.pcolormesh(X, Y, goodness, cmap=GOODNESS_CMAP, vmin=0, vmax=1,
                  shading="gouraud", zorder=0, alpha=0.6, rasterized=True)


# "Better" arrow endpoints (accuracy %, seconds). Routed through the empty band
# between the LLM cluster (10^2-10^4 s) and the deterministic points, clear of
# every marker and label.
BETTER_ARROW = ((3, 30), (86, 0.05))
BETTER_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("better", ["#9fd3e6", "#1f4e9c"])


def draw_better_arrow(ax):
    """Diagonal arrow, light to dark blue, pointing to the ideal corner, labelled "better"."""
    (x0, y0), (x1, y1) = BETTER_ARROW
    n = 80
    xs = np.linspace(x0, x1, n)
    ys = np.logspace(np.log10(y0), np.log10(y1), n)  # straight on the log axis
    points = np.column_stack([xs, ys])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap=BETTER_CMAP, linewidths=2.2, capstyle="round", zorder=3)
    line.set_array(np.linspace(0, 1, n - 1))
    ax.add_collection(line)
    ax.annotate("", xy=(x1, y1), xytext=(xs[-4], ys[-4]),
                arrowprops=dict(arrowstyle="-|>", color=BETTER_CMAP(1.0), linewidth=2.2,
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=3)
    # Label at the midpoint, rotated to follow the arrow as drawn on screen
    mid = n // 2
    (px0, py0), (px1, py1) = ax.transData.transform([points[mid - 5], points[mid + 5]])
    angle = np.degrees(np.arctan2(py1 - py0, px1 - px0))
    ax.annotate("better", xy=points[mid], xytext=(0, 5), textcoords="offset points",
                rotation=angle, rotation_mode="anchor", ha="center", va="bottom",
                fontsize=7, color="#1f4e9c", fontweight="bold", zorder=3)


def plot_efficiency_scatter(rows):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.85, 3.3))
    xlim, ylim = (-4, 106), (2e-3, 5e4)
    draw_goodness_gradient(ax, xlim, ylim)

    for row in rows:
        color = IDENTITY_COLOR[row["identity"]]
        direct, tools = row["direct"], row["tools"]
        if direct and tools:
            ax.annotate("", xy=(tools["accuracy"], tools["total_time_s"]),
                        xytext=(direct["accuracy"], direct["total_time_s"]),
                        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.8,
                                        shrinkA=4, shrinkB=4, mutation_scale=7, alpha=0.8),
                        zorder=4)
        for mode, point in (("direct", direct), ("tools", tools)):
            if point is None or row["identity"] in ("soap", "mcp"):
                continue  # SOAP and MCP sit on REST's point; labelled together below
            marker = "o" if mode == "direct" else "D"
            ax.scatter(point["accuracy"], point["total_time_s"], s=28 if mode == "direct" else 22,
                       marker=marker, facecolors=color, edgecolors="white", linewidths=0.6, zorder=5)
            if row["identity"] in SCATTER_LABELS:
                label, (dx, dy) = SCATTER_LABELS[row["identity"]]
                ax.annotate(label, xy=(point["accuracy"], point["total_time_s"]), xytext=(dx, dy),
                            textcoords="offset points", fontsize=6, color=INK_SECONDARY,
                            ha="left" if dx >= 0 else "right", va="center")

    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Total time (s, log scale)")
    ax.grid(True, which="major", color=GRID, linewidth=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_better_arrow(ax)

    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MUTED, markersize=5,
                   label="Deterministic / direct"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=MUTED, markersize=4.5,
                   label="LLM + MCP tools"),
        plt.Line2D([0], [0], color=MUTED, linewidth=0.8, label="Same model: direct → + tools"),
    ]
    model_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=IDENTITY_COLOR[identity],
                   markersize=5, label=label)
        for label, identity, *_ in LLM_MODELS
    ]
    # Horizontal key below the plot: marker meanings, then model colours.
    # Figure-level legends, so the tight crop keeps both.
    fig.legend(handles=marker_handles, loc="upper center", bbox_to_anchor=(0.53, 0.0),
               ncol=3, frameon=False, columnspacing=1.6, handletextpad=0.4)
    fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.53, -0.055),
               ncol=6, frameon=False, columnspacing=1.2, handletextpad=0.3)
    save(fig, "final_efficiency_scatter")


# ============================================================================
# Figure 3a — Outcome breakdown: stacked outcome shares per model, direct
# and + tools side by side.
# ============================================================================

SHORT_NAME = {
    "llama3.2:1b": "Llama\n1B", "llama3.2:3b": "Llama\n3B", "phi4": "Phi4\nmini",
    "granite": "Granite4\n3B", "qwen2": "Qwen2\nMath", "qwen3.5": "Qwen3.5\n4B",
}


def draw_outcome_breakdown(ax, rows):
    llm_rows = [r for r in rows if r["group"] == "LLM" and r["identity"] != "groq"]
    width, gap = 0.36, 0.04
    xticks, xlabels = [], []
    for i, row in enumerate(llm_rows):
        present = [m for m in ("direct", "tools") if row[m] is not None]
        offsets = [-(width + gap) / 2, (width + gap) / 2] if len(present) == 2 else [0]
        for mode, dx in zip(present, offsets):
            s = row[mode]
            total = s["correct"] + sum(s["failures"].values())
            bottom = 0
            for key in ("correct", "wrong_answer", "format_failure", "api_error"):
                share = 100 * (s["correct"] if key == "correct" else s["failures"][key]) / total
                style = {"facecolor": OUTCOME_COLOR[key], "edgecolor": SURFACE, "linewidth": 0.4}
                if mode == "tools":
                    style.update(hatch="//////", edgecolor="white")
                ax.bar(i + dx, share, bottom=bottom, width=width, zorder=3, **style)
                bottom += share
        xticks.append(i)
        xlabels.append(SHORT_NAME[row["identity"]])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=5.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of 1,000 equations (%)")
    style_axes(ax, grid_axis="y")


def outcome_legend_handles():
    handles = [mpatches.Patch(facecolor=OUTCOME_COLOR[k], label=OUTCOME_LABEL[k]) for k in OUTCOME_COLOR]
    handles.append(mpatches.Patch(facecolor=MUTED, edgecolor="white", hatch="//////", label="+ MCP tools"))
    return handles


# ============================================================================
# Figure 3b — Temperature effect: T=0 vs T=0.7 accuracy for every model and
# mode, as a dumbbell (filled = T=0, hollow = T=0.7). Short lines = little effect.
# ============================================================================

def draw_temperature_effect(ax, rows):
    llm_rows = [r for r in rows if r["group"] == "LLM" and r["identity"] != "groq"]
    y, ticks, labels = 0, [], []
    for row in llm_rows:
        for mode in ("direct", "tools"):
            s = row[mode]
            if s is None:
                continue
            color = IDENTITY_COLOR[row["identity"]]
            a0, a7 = s["by_temp"][0.0], s["by_temp"][0.7]
            marker = "o" if mode == "direct" else "D"
            ax.plot([a0, a7], [y, y], color=color, linewidth=1.2, alpha=0.5, zorder=3)
            # Hollow T=0.7 drawn larger and underneath, filled T=0 smaller on top,
            # so both stay visible when the two values nearly coincide
            ax.scatter([a7], [y], marker=marker, s=26, facecolors=SURFACE, edgecolors=color, linewidths=0.8, zorder=4)
            ax.scatter([a0], [y], marker=marker, s=9, facecolors=color, edgecolors=color, linewidths=0.4, zorder=5)
            ax.text(103, y, f"Δ {abs(a7 - a0):.1f}", va="center", fontsize=5.5, color=INK_SECONDARY)
            ticks.append(y)
            labels.append(f"{row['label']}" + ("  + tools" if mode == "tools" else ""))
            y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlim(-2, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Accuracy (%)")
    style_axes(ax, grid_axis="x")


def temperature_legend_handles():
    return [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=4.5, label="T = 0"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=4.5, label="T = 0.7"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=MUTED, markersize=4, label="+ MCP tools"),
    ]


def plot_outcome_breakdown(rows):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN / 2, 2.6))
    draw_outcome_breakdown(ax, rows)
    ax.legend(handles=outcome_legend_handles(), loc="upper center", bbox_to_anchor=(0.45, -0.2),
              ncol=3, frameon=False, fontsize=5.8, columnspacing=1.0, handlelength=1.4)
    save(fig, "final_outcome_breakdown")


def plot_temperature_effect(rows):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN / 2, 2.6))
    draw_temperature_effect(ax, rows)
    ax.legend(handles=temperature_legend_handles(), loc="upper center", bbox_to_anchor=(0.45, -0.16),
              ncol=3, frameon=False)
    save(fig, "final_temperature_effect")


def plot_outcome_distribution(rows):
    """Figure 3 in the paper: (a) outcome breakdown and (b) temperature effect."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7),
                                     gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.55})
    draw_outcome_breakdown(ax_a, rows)
    ax_a.set_title("(a) Outcome breakdown", loc="left", fontweight="bold", pad=4)
    ax_a.legend(handles=outcome_legend_handles(), loc="upper center", bbox_to_anchor=(0.45, -0.2),
                ncol=3, frameon=False, fontsize=5.8, columnspacing=1.0, handlelength=1.4)
    draw_temperature_effect(ax_b, rows)
    ax_b.set_title("(b) Temperature effect on accuracy", loc="left", fontweight="bold", pad=4)
    ax_b.legend(handles=temperature_legend_handles(), loc="upper center", bbox_to_anchor=(0.4, -0.16),
                ncol=3, frameon=False)
    save(fig, "final_outcome_distribution")


# ============================================================================
# Table 1 — results table as LaTeX, in the paper's original column layout
# (one row per temperature), split into three delineated sections:
# deterministic function calling, direct LLM invocation, and LLM + MCP tools.
# ============================================================================

def tex_int(v):
    """Thousands separator in the paper's style: 47{,}130."""
    return f"{v:,.0f}".replace(",", "{,}")


def table_row(label, r, base, cost=None, bold=False):
    """One LaTeX row: Acc, total time, avg ms per sample, Time×, process cost, Cost×."""
    total = r["total_time_s"]
    avg_ms = total / r.get("n", 1000) * 1000
    total_txt = f"{total:.4f}" if total < 0.1 else f"{total:.2f}"
    avg_txt = f"{avg_ms:.3f}" if avg_ms < 0.1 else f"{avg_ms:.2f}"
    if cost is None:
        cost_txt, cost_x = "n/a", "n/a"
    else:
        cost_txt = f"{cost:.4f}" if cost < 10 else f"{cost:.2f}"
        cost_x = f"{tex_int(cost / base['cost'])}$\\times$"
    if bold:
        total_txt, avg_txt, cost_txt = (f"\\textbf{{{t}}}" for t in (total_txt, avg_txt, cost_txt))
    cells = [label, f"{r['accuracy']:.1f}\\%" if r["accuracy"] != 100 else "100\\%",
             total_txt, avg_txt, f"{tex_int(total / base['time'])}$\\times$", cost_txt, cost_x]
    return " & ".join(cells) + " \\\\"


def section_header(title):
    return [f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\", "\\hline"]


def write_results_table(results):
    fc = results["vanilla"]
    base = {"time": fc["total_time_s"], "cost": cost_seconds(fc)}

    lines = section_header("Function Calling (deterministic)")
    for label, _, key in DETERMINISTIC:
        r = results[key]
        is_base = key == "vanilla"
        lines.append(table_row(label + ("*" if is_base else ""), r, base, cost_seconds(r), bold=is_base))

    lines.append("\\hline")
    lines += section_header("Direct invocation (LLM)")
    lines.append(table_row("API (gpt-oss-20b)$^\\dagger$", {**API_BASELINE, "n": 1000}, base))
    for label, _, model, has_direct, _ in LLM_MODELS:
        if has_direct:
            for t in TEMPERATURES:
                r = results[f"ollama_llm:{model}:T{t}"]
                lines.append(table_row(f"{label} (T={t:g})", r, base, cost_seconds(r)))

    lines.append("\\hline")
    lines += section_header("LLM + Tools (MCP)")
    for label, _, model, _, has_tools in LLM_MODELS:
        if has_tools:
            for t in TEMPERATURES:
                r = results[f"mcp_agent_ollama:{model}:T{t}"]
                lines.append(table_row(f"{label} (T={t:g})", r, base, cost_seconds(r)))

    tex = r"""\begin{table*}[t]
\centering
\begin{tabular}{lccccccc}
\hline
\small
\textbf{Method}  & \textbf{Acc. (\%)} & \textbf{T.Time(s)} & \textbf{Avg.Time/Smpl(ms)} & \textbf{Time$\times$} & \textbf{Process Cost(s)} & \textbf{Cost$\times$} \\
\hline
""" + "\n".join(lines) + r"""
\hline
\end{tabular}
\caption{Benchmark results across all evaluation methods on 1{,}000 equations. Times $\geq$1s are shown to two decimal places. \textbf{Time$\times$} and \textbf{Cost$\times$} report the slowdown/overhead factor relative to the Function Calling baseline (Total Time and Processor Runtime columns, respectively). For local Ollama models, Processor Runtime is Ollama's own self-reported inference compute time (prompt processing + generation duration) and reflects both CPU and GPU execution; for LLM + Tools it is summed over every LLM turn of an equation. \textbf{LLM + Tools} gives each model four MCP tools (add, subtract, multiply, divide) to call instead of computing the answer itself. Qwen2-Math-1.5B does not support tool calling and appears only under direct invocation; Qwen3.5-4B appears only with tools, as answering directly it frequently exceeded the 30\,s request timeout. API invocation cost 39{,}180 tokens in total (15{,}571 prompt + 23{,}609 completion). \textit{*Function Calling is the baseline against which the other methods are compared.}}
\label{tab:results}
\end{table*}
"""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "table_results.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Saved → {path}")


def main():
    results = load_results()
    rows = build_rows(results)
    plot_measures(rows)
    plot_efficiency_scatter(rows)
    plot_outcome_breakdown(rows)
    plot_temperature_effect(rows)
    plot_outcome_distribution(rows)
    write_results_table(results)


if __name__ == "__main__":
    main()
