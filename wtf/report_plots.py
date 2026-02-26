"""
AI generated Report plotting script for RL experiments.
prompted witht he desired outline of the plots and 

Usage:
- Set SAVE_DIR
- Set paths to .pth stats files
- Run: python report_plots.py
"""

import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# ============================================================
# USER CONFIG
# ============================================================

SAVE_DIR = "./plots"  # where figures are saved

REWARD_PATHS = [
    "checkpoints/2026-02-20_17-12-53-Hockey-SAC-critic-optimADAM-polic-optimADAM/stats_20000.pth",
    "checkpoints/2026-02-20_21-05-00-Hockey-SAC-critic-optimSLS-polic-optimADAM/stats_15250.pth",
    "checkpoints/2026-02-20_21-06-38-Hockey-SAC-critic-optimADAM-polic-optimSLS/stats_14750.pth",
    "checkpoints/2026-02-20_21-20-19-Hockey-SAC-critic-optimSLS-polic-optimSLS/stats_14750.pth",
    "checkpoints/2026-02-21_13-47-26-Hockey-DDPG-0.01/stats_11250.pth",
]
# First 4: different optimizer settings (consistent colors)
RUN_PATHS = [
    "checkpoints/2026-02-20_17-12-53-Hockey-SAC-critic-optimADAM-polic-optimADAM/stats_50000.pth",
    "checkpoints/2026-02-20_21-05-00-Hockey-SAC-critic-optimSLS-polic-optimADAM/stats_15250.pth",
    "checkpoints/2026-02-20_21-06-38-Hockey-SAC-critic-optimADAM-polic-optimSLS/stats_14750.pth",
    "checkpoints/2026-02-20_21-20-19-Hockey-SAC-critic-optimSLS-polic-optimSLS/stats_14750.pth",
]

# Next 4: same config, different seeds (SLS seeds for row 2 col 1)
SEED_PATHS = [
    "checkpoints/2026-02-20_21-20-19-Hockey-SAC-critic-optimSLS-polic-optimSLS/stats_14750.pth",
    "checkpoints/2026-02-20_17-27-21-Hockey-SAC-critic-optimSLS-polic-optimSLS/stats_5000.pth",
    "checkpoints/2026-02-21_11-18-04-Hockey-SAC-critic-optimSLS-polic-optimSLS/stats_12250.pth",
    "checkpoints/2026-02-21_11-18-05-Hockey-SAC-critic-optimSLS-polic-optimSLS/stats_12000.pth",
]
REWARD_LABELS=[
    "Adam-Adam",
    "Sls-Adam",
    "Adam-Sls",
    "Sls-Sls",
    "DDPG",
]
# Labels for legend (same order as RUN_PATHS)
RUN_LABELS = [
    "Adam-Adam",
    "Sls-Adam",
    "Adam-Sls",
    "Sls-Sls",
]

# ============================================================
# STYLE SETUP
# ============================================================

plt.rcParams.update({
    'figure.facecolor':     '#ffffff',
    'axes.facecolor':       '#fafafa',
    'axes.edgecolor':       '#cccccc',
    'axes.grid':            True,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.xmargin':         0.01,
    'axes.ymargin':         0.0,
    'grid.alpha':           0.3,
    'grid.color':           '#888888',
    'grid.linestyle':       '--',
    'font.family':          'sans-serif',
    'font.size':            9,
    'axes.titlesize':       12,
    #'axes.titleweight':     'bold',
    'axes.labelsize':       9,
    'legend.fontsize':      10,
    'legend.framealpha':    0.9,
    'legend.edgecolor':     '#cccccc',
    'figure.dpi':           130,
    'savefig.dpi':          200,
    'savefig.bbox':         'tight',
})

COLORS = ["#1A109E",  # dunkelblau
          "#4FC1E7",  # blue
          "#0884D7",  # green
          "#289003",  # purple
          "#FA5109", #orange
          "#46CAEF",  # türkis
          '#D4A843',  # yellow
          ]
COLORS0 = [
          "#1A109E",  # dunkelblau
          "#4FC1E7",  # blue
          "#0884D7",  # green
          "#289003",  # purple
          "#46CAEF",  # türkis
          '#D4A843',  # yellow
          ]
COLORSc = ["#4FC1E7",  # blue
          "#289003",  # purple
          ]
COLORSp = ["#0884D7",  # green
          "#289003",  # purple
          ]

REWARD_SMOOTH = 0.01
LOSS_SMOOTH = 0.001
LR_SMOOTH = 0.0005

def get_color(i, Colors =COLORS):
    return COLORS[i % len(Colors)]


# ============================================================
# UTILITIES
# ============================================================
def mea_smooth(x, alpha=0.03):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    sm = np.zeros_like(x)
    sm[0] = x[0]
    for i in range(1, len(x)):
        sm[i] = alpha * x[i] + (1 - alpha) * sm[i-1]
    return sm

def smooth(x, window):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def load_stats(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    #data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def lighten(color, factor=0.3):
    """Lighten color for seed variations."""
    c = np.array(to_rgb(color))
    return tuple(c + (1 - c) * factor)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_smoothed(ax, values, label, color, window):
    values = np.asarray(values)
    sm = mea_smooth(values, window)
    ax.plot(sm, label=label, color=color, linewidth=1, alpha=0.8)

def plot_with_variance(ax, values, label, color, alpha, shade_alpha=0.15):
    """
    EMA line + rolling std shading
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return

    # --- EMA mean ---
    ema = mea_smooth(values, alpha)

    # --- rolling std ---
    # window tied to smoothing strength
    win = max(10, int(1 / alpha))
    std = np.array([
        np.std(values[max(0, i - win):i + 1])
        for i in range(len(values))
    ])

    # --- Plot ---
    ax.plot(ema, label=label, color=color, linewidth=1.2, alpha=0.95)

    ax.fill_between(
        np.arange(len(ema)),
        ema - std,
        ema + std,
        color=color,
        alpha=shade_alpha,
        linewidth=0,
    )

def plot_lr_with_std(ax, lr_list, label, color, lr_smooth):
    arr = np.asarray(lr_list)
    if arr.ndim == 1:
        ax.plot(arr, label=label, color=color, linewidth=2)
        return

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ax.plot(mean, label=label, color=color, linewidth=2)
    ax.fill_between(np.arange(len(mean)), mean - std, mean + std,
                    color=color, alpha=0.2)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    runs = [load_stats(p) for p in RUN_PATHS]
    rewards = [load_stats(p) for p in REWARD_PATHS]
    seeds = [load_stats(p) for p in SEED_PATHS]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))

    # ========================================================
    # ROW 1 — Main comparisons
    # ========================================================

    # (1,1) Reward
    ax = axes[0, 0]
    print(len(rewards))
    print(len(REWARD_LABELS))
    for i, stats in enumerate(rewards):
        if stats is None:
            continue
        print(i)
        plot_with_variance(ax, stats["rewards"], REWARD_LABELS[i], get_color(i,COLORS0), REWARD_SMOOTH)
    ax.set_title("Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")

    # (1,2) Q loss
    ax = axes[0, 1]
    for i, stats in enumerate(runs):
        if stats is None:
            continue
        plot_with_variance(ax, stats["c_loss"], RUN_LABELS[i], get_color(i), LOSS_SMOOTH)
    ax.set_ylim(-1,15)
    ax.set_title("Critic Loss")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Q loss")

    # (1,3) Critic LR
    ax = axes[0, 2]
    for i, stats in enumerate(runs):
        if stats is None:
            continue
        if i ==1:
            plot_smoothed(ax, stats["critic_lrs"], RUN_LABELS[i], get_color(i, COLORS), LR_SMOOTH)
        elif i ==3:
            plot_smoothed(ax, stats["critic_lrs"], RUN_LABELS[i], get_color(i, COLORS), LR_SMOOTH)
        else:
            pass
    ax.set_title("Critic LR")
    ax.set_yscale('log')
    ax.set_xlabel("Updates")
    ax.set_ylabel("LR")

    # ========================================================
    # ROW 2
    # ========================================================

    # (2,1) Multiple SLS seeds reward (same color tones)
    ax = axes[1, 0]
    base_color = get_color(3)
    c =["#289003", "#65B81C", "#97CA5C", "#A1CA92"]
    for i, stats in enumerate(seeds):
        if stats is None:
            continue
        #c = lighten(base_color, factor=0.2 * i)
        plot_with_variance(ax, stats["rewards"], f"Seed {i+1}", c[i], REWARD_SMOOTH)
    ax.set_title("Sls Seeds — Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")

    # (2,2) Policy loss
    ax = axes[1, 1]
    for i, stats in enumerate(runs):
        if stats is None:
            continue
        plot_with_variance(ax, stats["p_loss"], RUN_LABELS[i], get_color(i), 0.006)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Updates")
    ax.set_ylabel("Loss")

    # (2,3) Policy LR
    ax = axes[1, 2]
    for i, stats in enumerate(runs):
        if stats is None:
            continue
        elif i == 2: 
            plot_smoothed(ax, stats["policy_lrs"], RUN_LABELS[i], get_color(i), LR_SMOOTH)
        elif i ==3: 
            plot_smoothed(ax, stats["policy_lrs"], RUN_LABELS[i], get_color(i), LR_SMOOTH)
    ax.set_title("Policy LR")
    ax.set_yscale('log')
    ax.set_xlabel("Updates")
    ax.set_ylabel("LR")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.12),
                   ncol=min(len(labels), 5),
                   frameon=True, 
                   title = "SAC with (Critic-Actor) optimizer"),
    
        #fig.update_layout(legend_title ="SAC with (Critic-Actor)")
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, "report_training_plots.png")
    out_path_pdf = os.path.join(SAVE_DIR, "report_training_plots.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(out_path_pdf, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
