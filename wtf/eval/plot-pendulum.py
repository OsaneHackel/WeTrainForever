'''
AI generated plotting script for comparing reward curves of different agents on the Pendulum environment.
'''

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


SAVE_DIR = "./plots"

RUN_PATHS = [
    "checkpoints/Pendulum_DDPG/km3GCms_DDPG-stat.pkl",
    "checkpoints/Pendulum_SAC/78DMCB8_SLS-stat.pkl",
    "checkpoints/Pendulum_SAC/tb2kHmU_ADAM-stat.pkl",
    "checkpoints/Pendulum_TD3/TD3_Pendulum-v1-noiseGaussian-eps0.1-s0-stat.pkl", 
    "checkpoints/Pendulum_TD3/TD3_Pendulum-v1-noiseOrnsteinU-eps0.1-s0-stat.pkl",
    "checkpoints/Pendulum_TD3/TD3_Pendulum-v1-noisePink-eps0.1-s0-stat.pkl"
]

LABELS = ["DDPG", "SAC (Sls)", "SAC (Adam)", "TD3 (Gaussian)", "TD3 (OU)", "TD3 (Pink)"] 


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

COLORS = ['#2176AE', '#57A773', '#D4A843', '#E05A3A', '#8B5FBF', '#E87EA1']

LINE_ALPHA = 0.75
SMOOTHING_WINDOW = 200


def get_color(i):
    return COLORS[i % len(COLORS)]

def smooth(x, window=SMOOTHING_WINDOW):
    x = np.asarray(x)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def load_stats(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    runs = [load_stats(p) for p in RUN_PATHS]

    plt.figure(figsize=(5, 3))

    for i, stats in enumerate(runs):
        if stats is None:
            continue

        label = LABELS[i] if LABELS else f"Run {i+1}"
        rewards = stats["rewards"]

        sm = smooth(rewards)
        plt.plot(sm, label=label,
                 color=get_color(i),
                 alpha=LINE_ALPHA,
                 linewidth=2)

    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, "pendulum_rewards.pdf")
    plt.savefig(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
