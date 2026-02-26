import numpy as np
import random
from pathlib import Path
import imageio
import hockey.hockey_env as h_env
import matplotlib.pyplot as plt

from wtf.utilssac import load_agent
from Hockey_TD3.TD3_agent import TD3_Agent
from gymnasium import spaces

#plotting parameters
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
# ==============================
# AGENT LOADERS
# ==============================

def load_sac(path):
    return load_agent(path, "SAC", evaluate=True)

def load_td3(path):
    env = h_env.HockeyEnv()
    full_action_space = env.action_space
    n_per_player = full_action_space.shape[0] // 2

    agent = TD3_Agent(
        obs_dim=env.observation_space.shape[0],
        observation_space=env.observation_space,
        act_dim=n_per_player,
        action_space=spaces.Box(-1, 1, (n_per_player,), dtype=np.float32),
    )
    agent.load(path)
    return agent

def load_basic(weak=True):
    return h_env.BasicOpponent(weak=weak)

# ==============================
# UNIFIED ACTION INTERFACE
# ==============================

def act(agent, obs):
    if hasattr(agent, "act"):
        try:
            return agent.act(obs, eps=0.0)
        except TypeError:
            return agent.act(obs)
    if hasattr(agent, "select_action"):
        return agent.select_action(obs, explore=False)
    raise ValueError(f"Unknown agent type {type(agent)}")

# ==============================
# GAME SIMULATION
# ==============================

def play_episode(agent_a, agent_b, render=False):
    env = h_env.HockeyEnv()
    env.one_starts = random.random() > 0.5

    obs_a, _ = env.reset()
    obs_b = env.obs_agent_two()

    frames = []

    for _ in range(250):
        a1 = act(agent_a, obs_a)
        a2 = act(agent_b, obs_b)

        obs_next, _, done, trunc, info_a = env.step(np.hstack([a1, a2]))
        info_b = env.get_info_agent_two()

        if render:
            frames.append(env.render(mode="rgb_array"))

        obs_a = obs_next
        obs_b = env.obs_agent_two()

        if done or trunc:
            break

    return info_a["winner"], frames


# ==============================
# WINRATE WITH SIDE SWAP
# ==============================

def winrate(agent_a, agent_b, episodes=200):
    wins = 0
    draws = 0

    for ep in range(episodes):
        swap = (ep % 2 == 1)

        if not swap:
            winner, _ = play_episode(agent_a, agent_b)
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
        else:
            winner, _ = play_episode(agent_b, agent_a)
            if winner == -1:
                wins += 1
            elif winner == 0:
                draws += 1

    return {
        "winrate": wins / episodes,
        "drawrate": draws / episodes,
        "wins": wins,
        "episodes": episodes,
    }


# ==============================
# GIF GENERATION
# ==============================

def save_gif(agent_a, agent_b, out_path, name):
    _, frames = play_episode(agent_a, agent_b, render=True)
    frames = 10 * frames[0:1] + frames  # loop back to start for smoothness
    imageio.mimwrite(out_path / f"{name}.gif", frames, fps=30, loop = 0)

def plot_winrates(results: dict, out_path, title="Agent Comparison"):
    """
    Stacked horizontal bars:
    Green = wins, Yellow = draws, Red = losses
    """

    labels = []
    wins = []
    draws = []
    losses = []

    for matchup, stats in results.items():
        labels.append(matchup.replace("_vs_", " vs "))

        win = stats["winrate"]
        draw = stats.get("drawrate", 0.0)
        loss = 1.0 - win - draw

        wins.append(win)
        draws.append(draw)
        losses.append(loss)

    # Sort by winrate (nice visual ordering)
    order = sorted(range(len(wins)), key=lambda i: wins[i])
    labels = [labels[i] for i in order]
    wins = [wins[i] for i in order]
    draws = [draws[i] for i in order]
    losses = [losses[i] for i in order]

    # Colors (paper friendly)
    win_color = "#4CAF50"   # green
    draw_color = "#FFC107"  # yellow
    loss_color = "#F44336"  # red

    plt.figure(figsize=(4, 3))

    # Stacked bars
    plt.barh(labels, losses, color=loss_color, label="Loss")
    plt.barh(labels, draws, left=losses, color=draw_color, label="Draw")
    plt.barh(labels, wins, left=[l + d for l, d in zip(losses, draws)],
             color=win_color, label="Win")

    # Annotate winrates
    for i, (w, d, l) in enumerate(zip(wins, draws, losses)):
        plt.text(1.02, i, f"{w:.2f}", va="center", fontsize=9)

    plt.xlabel("Outcome ratio")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # center below plot
        ncol=3  # horizontal legend
    )

    # Clean look (paper style)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save high-quality outputs
    png_path = out_path / "winrate_stacked.png"
    pdf_path = out_path / "winrate_stacked.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)  # vector for papers
    plt.close()

    print(f"Saved plots → {png_path} and {pdf_path}")

# ==============================
# COMPARISON MATRIX
# ==============================

def compare_all(sac_path, td3_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load agents
    sac = load_sac(sac_path)
    td3 = load_td3(td3_path)
    weak = load_basic(True)
    strong = load_basic(False)

    agents = {
        "SAC": sac,
        "TD3": td3,
        "weak": weak,
        "strong": strong,
    }

    names = list(agents.keys())

    results = {}

    # Pairwise comparisons
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            a, b = agents[name_a], agents[name_b]

            print(f"\n=== {name_a} vs {name_b} ===")

            stats = winrate(a, b)
            results[f"{name_a}_vs_{name_b}"] = stats

            print(stats)

            # Save GIFs
            save_gif(a, b, out_dir, f"{name_a}_vs_{name_b}")
            save_gif(b, a, out_dir, f"{name_b}_vs_{name_a}")

    # Save results
    with open(out_dir / "results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    print("\nSaved results to", out_dir)
    plot_winrates(results, out_dir)

import ast

def load_results_txt(path):
    results = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            name, data = line.split(":", 1)
            results[name.strip()] = ast.literal_eval(data.strip())
    return results

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    sac_path = Path("BestAgents/SAC_best_3.pth")
    td3_path = Path("BestAgents/TD3_best.pt")
    out_dir = Path("plots/comparison_results")

    compare_all(sac_path, td3_path, out_dir)
    #results = load_results_txt(out_dir/"results.txt")
    #plot_winrates(results, out_dir)