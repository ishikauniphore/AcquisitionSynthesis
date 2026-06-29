"""
Plots for multilingual eval results.
3 models x 4 evaluations x 3 metrics x (2-3 methods).

Each cell in the source tables is a triplet: (EmbedSim, ROUGE-L, LAJ).
One figure per evaluation type. Each figure is a grid:
    rows    = metric  (EmbedSim, ROUGE-L, LAJ)
    columns = model   (Qwen 7B, Qwen 14B, Llama 8B)
Each subplot is a grouped bar chart: x = categories (languages or task),
bars grouped by method (Base, DataEnvGym, Original).
Qwen 14B has no Original yet -> only 2 bars.

Y-axis is truncated per metric-row (shared across the 3 models in that row)
so small differences are visible; true values are printed on each bar.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------- config
SHOW_VALUES   = True       # print the true number on top of each bar
VALUE_FMT     = "{:.1f}"   # label format
# Y-axis floor behaviour:
#   "adaptive" -> zoom each metric-row to its own data range (recommended)
#   a number   -> hard floor everywhere (e.g. 40); bars below it are clipped
Y_FLOOR_MODE  = "adaptive"
# -----------------------------------------------------------------------

METRICS = ["EmbedSim", "ROUGE-L", "LAJ"]
METHODS = ["Base", "DataEnvGym", "Original"]
MODELS  = ["Qwen 7B", "Qwen 14B", "Llama 8B"]
EVALS   = ["Translation (OPUS)", "Factual Knowledge (MMMLU)",
           "RAG (mHotPotQA)", "Agentic"]

METHOD_COLORS = {
    "Base":       "#6b7280",   # slate gray
    "DataEnvGym": "#e07b39",   # warm orange
    "Original":   "#3b7a57",   # deep green
}

# Each entry: list of [EmbedSim, ROUGE-L, LAJ] per category, in the order of `cats`.
DATA = {
"Qwen 7B": {
  "Translation (OPUS)": {
    "cats": ["Arabic","French","German","Italian","Japanese","Portuguese","Spanish"],
    "Base":      [[73.4,45.0,45.6],[85.0,63.8,72.6],[77.8,51.2,56.6],[86.2,58.0,64.6],[66.0,23.6,39.8],[87.2,63.4,66.2],[85.4,65.8,71.6]],
    "DataEnvGym":[[52.2,30.8,44.8],[54.4,38.6,70.0],[56.0,35.0,54.8],[65.6,42.6,62.6],[50.6,17.8,36.2],[70.2,48.6,65.4],[65.2,46.8,68.0]],
    "Original":  [[71.4,42.8,47.2],[79.4,60.0,70.6],[72.2,47.0,56.4],[80.6,54.0,64.4],[65.8,21.8,35.8],[85.2,58.6,67.2],[85.0,64.0,70.6]],
  },
  "Factual Knowledge (MMMLU)": {
    "cats": ["Arabic","German","Spanish","French","Italian","Japanese","Portuguese"],
    "Base":      [[62.0,62.0,62.8],[69.2,69.2,69.4],[68.2,68.2,69.4],[70.8,70.8,72.0],[70.4,70.4,71.2],[61.4,61.4,64.0],[71.6,71.6,71.8]],
    "DataEnvGym":[[61.6,61.6,62.2],[67.8,67.8,67.6],[67.0,67.0,69.2],[70.0,70.0,71.6],[68.4,68.4,69.8],[60.8,60.8,61.8],[68.2,68.2,69.2]],
    "Original":  [[62.8,63.0,62.8],[69.8,69.8,70.0],[69.0,69.0,70.8],[74.4,74.4,76.4],[68.8,68.8,70.4],[62.2,62.2,62.2],[69.4,69.4,70.0]],
  },
  "RAG (mHotPotQA)": {
    "cats": ["Arabic","English","Russian","Chinese"],
    "Base":      [[28.8,3.8,8.0],[48.2,37.6,59.2],[33.0,8.4,23.6],[30.2,8.8,23.6]],
    "DataEnvGym":[[53.2,3.4,15.0],[73.6,63.0,63.0],[56.6,9.4,34.6],[51.0,8.0,27.8]],
    "Original":  [[31.3,3.5,10.8],[48.6,38.8,58.2],[33.8,8.8,30.8],[32.0,7.2,23.0]],
  },
  "Agentic": {
    "cats": ["Stem","Math","Chat"],
    "Base":      [[59.8,59.8,61.2],[51.8,25.2,15.8],[50.6,2.2,88.2]],
    "DataEnvGym":[[59.8,59.8,61.6],[47.6,20.2,13.4],[32.2,1.2,89.4]],
    "Original":  [[59.6,59.6,60.4],[51.0,23.4,13.6],[47.0,1.8,90.6]],
  },
},
"Qwen 14B": {
  "Translation (OPUS)": {
    "cats": ["Arabic","French","German","Italian","Japanese","Portuguese","Spanish"],
    "Base":      [[81.0,55.4,52.2],[80.2,58.0,72.0],[76.4,50.6,57.8],[84.6,57.2,64.6],[68.4,24.2,43.0],[84.8,62.0,67.2],[88.2,68.8,72.2]],
    "DataEnvGym":[[71.6,44.6,48.4],[74.0,49.8,70.8],[69.0,43.4,56.8],[76.0,46.4,62.6],[62.8,23.6,41.6],[76.6,53.4,67.4],[75.4,57.0,69.0]],
  },
  "Factual Knowledge (MMMLU)": {
    "cats": ["Arabic","German","Spanish","French","Italian","Japanese","Portuguese"],
    "Base":      [[70.0,70.0,70.0],[75.0,75.0,75.2],[74.2,74.2,75.0],[77.6,77.6,79.2],[72.2,72.2,73.4],[70.2,70.2,71.6],[73.8,73.8,75.0]],
    "DataEnvGym":[[66.2,66.2,66.6],[73.6,73.6,74.6],[71.4,71.4,72.0],[72.8,72.8,75.4],[71.2,71.2,73.4],[71.2,71.2,73.0],[72.6,72.6,75.4]],
  },
  "RAG (mHotPotQA)": {
    "cats": ["Arabic","English","Russian","Chinese"],
    "Base":      [[49.8,3.4,18.6],[67.6,57.6,63.0],[51.4,10.0,37.2],[48.2,8.8,30.0]],
    "DataEnvGym":[[48.4,3.4,19.2],[64.0,55.0,63.4],[50.6,9.8,37.0],[48.6,8.2,29.6]],
  },
  "Agentic": {
    "cats": ["Stem","Math","Chat"],
    "Base":      [[68.2,68.2,68.6],[51.8,27.6,17.0],[48.8,2.0,89.2]],
    "DataEnvGym":[[66.4,66.4,69.6],[51.6,26.4,15.2],[44.6,2.0,86.8]],
  },
},
"Llama 8B": {
  "Translation (OPUS)": {
    "cats": ["Arabic","French","German","Italian","Japanese","Portuguese","Spanish"],
    "Base":      [[50.8,29.8,44.8],[53.8,37.2,69.2],[52.6,30.8,53.4],[52.4,31.2,61.0],[38.6,14.2,35.4],[56.2,35.8,64.6],[57.0,42.2,70.0]],
    "DataEnvGym":[[37.8,20.2,40.2],[49.0,31.4,69.6],[42.8,23.6,53.2],[45.4,27.6,57.4],[32.8,9.2,33.6],[48.6,31.2,64.0],[45.6,31.0,67.6]],
    "Original":  [[39.6,20.6,42.8],[50.0,32.8,69.6],[47.0,27.6,54.0],[43.6,26.0,61.6],[34.4,11.4,34.8],[51.0,32.8,66.8],[53.4,36.4,68.2]],
  },
  "Factual Knowledge (MMMLU)": {
    "cats": ["Arabic","German","Spanish","French","Italian","Japanese","Portuguese"],
    "Base":      [[41.4,41.4,43.6],[53.0,53.0,56.8],[50.0,50.0,53.6],[53.2,53.2,58.0],[50.6,50.6,56.2],[45.4,45.4,48.0],[53.2,53.2,56.6]],
    "DataEnvGym":[[34.2,34.6,34.4],[44.6,44.6,50.4],[45.2,45.2,50.4],[50.6,50.8,57.8],[47.0,47.0,51.6],[40.4,40.4,45.0],[49.8,50.0,53.8]],
    "Original":  [[41.8,41.8,42.6],[52.0,52.0,54.0],[51.8,51.8,55.2],[53.2,53.2,57.8],[50.6,50.6,55.0],[41.4,41.4,44.4],[54.0,54.0,58.8]],
  },
  "RAG (mHotPotQA)": {
    "cats": ["Arabic","English","Russian","Chinese"],
    "Base":      [[57.0,2.6,21.0],[61.6,52.2,53.4],[57.8,9.4,36.6],[55.6,6.8,27.4]],
    "DataEnvGym":[[62.6,2.8,19.4],[66.8,50.2,51.2],[65.8,9.6,41.0],[58.6,8.2,30.0]],
    "Original":  [[61.4,3.2,22.0],[66.2,52.2,54.2],[61.4,10.4,39.8],[59.2,8.4,28.0]],
  },
  "Agentic": {
    "cats": ["Stem","Math","Chat"],
    "Base":      [[55.8,56.0,59.0],[26.2,8.8,4.4],[23.4,1.0,88.2]],
    "DataEnvGym":[[52.4,52.4,57.0],[31.2,9.4,4.0],[15.2,0.8,88.2]],
    "Original":  [[55.8,56.0,59.6],[26.8,7.4,3.8],[18.0,0.6,87.8]],
  },
},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})


def methods_for(model, eval_name):
    block = DATA[model][eval_name]
    return [m for m in METHODS if m in block]


def row_ylim(eval_name, metric_idx):
    """Shared y-limits for one metric-row, across all 3 models."""
    vals = []
    for model in MODELS:
        block = DATA[model][eval_name]
        for m in methods_for(model, eval_name):
            vals += [row[metric_idx] for row in block[m]]
    vmin, vmax = min(vals), max(vals)
    if Y_FLOOR_MODE != "adaptive":
        lo = float(Y_FLOOR_MODE)
        hi = min(102, vmax + max(4, 0.18 * (vmax - lo)))
        return lo, hi
    rng = vmax - vmin
    pad_below = max(2.0, 0.25 * rng)     # keep smallest bar ~visible
    pad_above = max(5.0, 0.20 * rng)     # headroom for value labels
    lo = max(0.0, math.floor((vmin - pad_below) / 5) * 5)
    hi = min(102.0, math.ceil((vmax + pad_above) / 5) * 5)
    return lo, hi


def draw_subplot(ax, eval_name, model, metric_idx, ylim, label_rot):
    block = DATA[model][eval_name]
    cats = block["cats"]
    methods_present = methods_for(model, eval_name)
    n_groups, n_bars = len(cats), len(methods_present)
    x = np.arange(n_groups)
    total_w = 0.80
    bar_w = total_w / n_bars
    lo, hi = ylim
    offset_txt = 0.012 * (hi - lo)

    for j, method in enumerate(methods_present):
        vals = [block[method][c][metric_idx] for c in range(n_groups)]
        offset = (j - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.4, zorder=3)
        if SHOW_VALUES:
            for xi, v in zip(x + offset, vals):
                ax.text(xi, min(v + offset_txt, hi), VALUE_FMT.format(v),
                        ha="center", va="bottom", rotation=label_rot,
                        fontsize=6 if n_groups > 4 else 7.5, color="#333333",
                        zorder=4)

    ax.set_xticks(x)
    rot = 35 if n_groups > 4 else 0
    ax.set_xticklabels(cats, rotation=rot, ha="right" if rot else "center", fontsize=9)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def make_figure(eval_name):
    wide = any(len(DATA[m][eval_name]["cats"]) > 4 for m in MODELS)
    figsize = (17, 12.5) if wide else (13.5, 11)
    label_rot = 90 if wide else 0
    fig, axes = plt.subplots(len(METRICS), len(MODELS), figsize=figsize)

    for r, metric in enumerate(METRICS):
        ylim = row_ylim(eval_name, r)
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            draw_subplot(ax, eval_name, model, r, ylim, label_rot)
            if r == 0:
                ax.set_title(model, fontsize=14, fontweight="bold", pad=10)

    for r, metric in enumerate(METRICS):
        pos = axes[r, 0].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, metric, va="center", ha="center",
                 rotation=90, fontsize=13, fontweight="bold", color="#222222")

    handles = [Patch(facecolor=METHOD_COLORS[m], label=m) for m in METHODS]
    fig.legend(handles, METHODS, loc="upper center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.965))

    fig.suptitle(eval_name, fontsize=18, fontweight="bold", y=0.998)
    fig.text(0.5, 0.004,
             "Score (0–100); y-axis truncated per row to show differences. "
             "Qwen 14B has no Original results yet (run in progress).",
             ha="center", fontsize=10, color="#666666")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.07,
                        hspace=0.48, wspace=0.18)
    return fig


fname_map = {
    "Translation (OPUS)": "translation_opus",
    "Factual Knowledge (MMMLU)": "factual_knowledge_mmmlu",
    "RAG (mHotPotQA)": "rag_mhotpotqa",
    "Agentic": "agentic",
}

from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages("/home/claude/all_eval_plots.pdf")
for ev in EVALS:
    fig = make_figure(ev)
    out = f"/home/claude/{fname_map[ev]}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)
pdf.close()
print("saved /home/claude/all_eval_plots.pdf")
