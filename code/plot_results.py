"""
Regenerate comparison plots from results/latest.json
Produces:
  - results/accuracy_vs_time_<ts>.png  — dual bar chart (accuracy %, time s)
  - results/scatter_<ts>.png           — scatter (accuracy vs total time)
  - results/raw_units_<ts>.png         — CPU-seconds, memory (MB), tokens — all in real units, no percentages
  - results/spider_<ts>.png            — radar chart, fixed-baseline 0-100 scores

All charting lives here, separate from benchmark.py — running a benchmark
never produces a plot as a side effect; this is a deliberate, later step.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ============================================================================
# Fixed reference baselines for the spider chart.
# These are absolute anchors, not "best/worst of this run" — that keeps a
# method's score meaning the same thing across separate runs, even if a
# method is skipped (e.g. a server wasn't up) in one of them.
# ============================================================================
SPEED_REFERENCE_S_PER_SAMPLE = 0.35   # 0 score at/above this, 100 score at 0s
MEM_REFERENCE_MB = 200.0              # 0 score at/above this, 100 score at 0MB
TOKENS_REFERENCE_PER_SAMPLE = 50.0    # 0 score at/above this, 100 score at 0 tokens


def load(path=None):
    path = path or os.path.join(RESULTS_DIR, "latest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_dual_bar(results, output_path):
    """Side-by-side bar chart: accuracy (%) and total time (s) per method."""
    methods = [r["method"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    times = [r["total_time_s"] for r in results]

    x = np.arange(len(methods))
    colors = plt.cm.tab10.colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    bars1 = ax1.bar(x, accuracy, color=[colors[i % len(colors)] for i in range(len(methods))], width=0.55, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy by Method", fontsize=13, fontweight="bold")
    ax1.axhline(100, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars1, accuracy):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # --- Total Time (log scale because vanilla is ~0.008s vs LLMs ~310s) ---
    bars2 = ax2.bar(x, times, color=[colors[i % len(colors)] for i in range(len(methods))], width=0.55, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax2.set_yscale("log")
    ax2.set_ylabel("Total Time (s)  [log scale]", fontsize=11)
    ax2.set_title("Total Time by Method", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f"{val:.2f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Method Comparison — Accuracy & Time", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dual bar chart saved → {output_path}")


def plot_scatter(results, output_path):
    """
    Scatter plot: x = accuracy (%), y = total time (s, log scale).
    Top-left corner = ideal (fast & accurate).
    """
    methods = [r["method"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    times = [r["total_time_s"] for r in results]
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (m, acc, t) in enumerate(zip(methods, accuracy, times)):
        c = colors[i % len(colors)]
        ax.scatter(acc, t, s=120, color=c, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(
            m,
            xy=(acc, t),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=10,
            color=c,
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("Total Time (s)  [log scale]", fontsize=12)
    ax.set_title("Accuracy vs Total Time\n(top-left = ideal)", fontsize=13, fontweight="bold")
    ax.set_xlim(-5, 110)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Annotate ideal corner
    ax.annotate("← ideal", xy=(0, min(times)), fontsize=9, color="grey",
                xytext=(5, 5), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved → {output_path}")


def plot_raw_units(results, output_path):
    """
    Everything that was previously folded into the spider chart's 0-100
    scores, shown instead in its own real, standardized unit:
      - CPU: CPU-seconds (exact user+sys time consumed, not sampled %)
      - Memory: MB (avg resident set size)
      - Tokens: total tokens used (LLM methods only; 0 for others)
    No axis here is a normalized/relative score — every bar is a real unit.
    """
    methods = [r["method"] for r in results]
    cpu_seconds = [r.get("cpu_seconds", r.get("client", {}).get("cpu_seconds", 0)) for r in results]
    mem_mb = [r["avg_memory_mb"] for r in results]
    tokens = [r.get("tokens", {}).get("total_tokens", 0) for r in results]

    x = np.arange(len(methods))
    colors = plt.cm.tab10.colors
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, values, title, ylabel, fmt in [
        (axes[0], cpu_seconds, "CPU Time Consumed", "CPU-seconds (user+sys)", "{:.3f}s"),
        (axes[1], mem_mb, "Memory Usage", "Avg RSS (MB)", "{:.1f} MB"),
        (axes[2], tokens, "Token Usage", "Total tokens", "{:d}"),
    ]:
        bars = ax.bar(x, values, color=bar_colors, width=0.55, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, values):
            label = fmt.format(int(val)) if fmt == "{:d}" else fmt.format(val)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                     ha="center", va="bottom", fontsize=8)

    fig.suptitle("Method Comparison — Raw Units (no percentages)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Raw-units chart saved → {output_path}")


def normalize_scores(results):
    """
    Convert raw metrics into 0-100 scores on five axes, each against a FIXED
    reference point (not "best/worst of this run") so scores stay comparable
    across separate runs:
      1. Accuracy        — already 0-100%, used directly.
      2. Speed           — 100 at 0s/sample, 0 at/above SPEED_REFERENCE_S_PER_SAMPLE.
      3. CPU Efficiency  — 100 at 0% CPU, 0 at/above 100% CPU (percent is already 0-100 by definition).
      4. Memory Efficiency — 100 at 0MB, 0 at/above MEM_REFERENCE_MB.
      5. Token Efficiency  — 100 at 0 tokens/sample, 0 at/above TOKENS_REFERENCE_PER_SAMPLE; non-LLM methods score 100.
    """
    scores = {}
    for r in results:
        accuracy = r["accuracy"]

        time_per_sample = r["avg_time_per_sample_s"]
        speed = 100.0 * (1 - min(time_per_sample / SPEED_REFERENCE_S_PER_SAMPLE, 1))

        cpu_pct = min(r["avg_cpu_percent"], 100.0)
        cpu_eff = 100.0 - cpu_pct

        mem_mb = r["avg_memory_mb"]
        mem_eff = 100.0 * (1 - min(mem_mb / MEM_REFERENCE_MB, 1))

        tokens_total = r.get("tokens", {}).get("total_tokens", 0)
        tokens_per_sample = tokens_total / r["n"] if r["n"] > 0 else 0
        tok_eff = 100.0 * (1 - min(tokens_per_sample / TOKENS_REFERENCE_PER_SAMPLE, 1))

        scores[r["method"]] = [
            round(accuracy, 2),
            round(speed, 2),
            round(cpu_eff, 2),
            round(mem_eff, 2),
            round(tok_eff, 2),
        ]

    return scores


def plot_spider(scores, output_path):
    """Generate and save a spider/radar chart comparing all methods."""
    categories = ["Accuracy", "Speed", "CPU\nEfficiency", "Memory\nEfficiency", "Token\nEfficiency"]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10.colors

    for idx, (method, values) in enumerate(scores.items()):
        vals = values + values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=8, color="grey")
    ax.set_title("Method Comparison (Spider Plot, fixed baselines)", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=10)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spider plot saved → {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load()
    results = data["results"]
    ts = data["timestamp"]

    print(f"\nLoaded {len(results)} methods from run {ts} (n={data['n']})\n")

    plot_dual_bar(results, os.path.join(RESULTS_DIR, f"accuracy_vs_time_{ts}.png"))
    plot_scatter(results, os.path.join(RESULTS_DIR, f"scatter_{ts}.png"))
    plot_raw_units(results, os.path.join(RESULTS_DIR, f"raw_units_{ts}.png"))

    if len(results) >= 2:
        scores = normalize_scores(results)
        plot_spider(scores, os.path.join(RESULTS_DIR, f"spider_{ts}.png"))
    else:
        print("[INFO] Need at least 2 methods to generate a spider plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
