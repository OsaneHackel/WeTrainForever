import random
import numpy as np
import pickle
import torch
import imageio
from pathlib import Path

import hockey.hockey_env as h_env

import wtf.plotting as plo
from wtf.agents.DDPG import DDPGAgent
from wtf.utilssac import load_agent

#TODO: use eps=0.0 for the evaluation

def simulate1(checkpoint_path, save_path, which_agent, suffix = None):
    env = h_env.HockeyEnv()
    env.one_starts = random.random() > 0.5
    agent = load_agent(checkpoint_path, which_agent, evaluate=True)
    opponent = h_env.BasicOpponent(weak=False)
    obs_opponent, _ = env.reset()
    obs_ddpg = env.obs_agent_two()
    frames = []
    for t in range(250):
        a_op = opponent.act(obs_opponent)
        a_ddpg =agent.act(obs_ddpg, eps=0.0)
        obs_next, r, done, trunc, _ = env.step(np.hstack([a_op, a_ddpg]))
        frames.append(env.render(mode='rgb_array'))
        obs_opponent = obs_next
        obs_ddpg = env.obs_agent_two()
        if done or trunc:
            break
    if suffix is not None:
        outpath = save_path / f"simulation_blue_{suffix}.gif"
    else:
        outpath = save_path / f"simulation_blue.gif"
    imageio.mimwrite(outpath, frames, fps=30)

def simulate2(checkpoint_path, save_path,which_agent, suffix = None):
    env = h_env.HockeyEnv()
    env.one_starts = random.random() > 0.5
    agent = load_agent(checkpoint_path, which_agent, evaluate=True)
    opponent = h_env.BasicOpponent(weak=False)
    obs_ddpg, _ = env.reset()
    obs_opponent = env.obs_agent_two()
    frames = []
    for t in range(250):
        a_ddpg =agent.act(obs_ddpg, eps=0.0)
        a_op = opponent.act(obs_opponent)
        obs_next, r, done, trunc, _ = env.step(np.hstack([a_ddpg, a_op]))
        frames.append(env.render(mode='rgb_array'))
        obs_ddpg = obs_next
        obs_opponent = env.obs_agent_two()
        if done or trunc:
            break
    if suffix is not None:
        outpath = save_path / f"simulation_red_{suffix}.gif"
    else:
        outpath = save_path / f"simulation_red.gif"
    imageio.mimwrite(outpath, frames, fps=30)

def simulate_selfplay(checkpoint_path, save_path,which_agent, suffix = None):
    env = h_env.HockeyEnv()
    env.one_starts = random.random() > 0.5
    agent = load_agent(checkpoint_path, which_agent, evaluate=True)
    opponent = load_agent(checkpoint_path, which_agent, evaluate=True)
    obs_opponent, _ = env.reset()
    obs_ddpg = env.obs_agent_two()
    frames = []
    for t in range(250):
        a_op = opponent.act(obs_opponent, eps=0.0)
        a_ddpg =agent.act(obs_ddpg, eps=0.0)
        obs_next, r, done, trunc, _ = env.step(np.hstack([a_op, a_ddpg]))
        frames.append(env.render(mode='rgb_array'))
        obs_opponent = obs_next
        obs_ddpg = env.obs_agent_two()
        if done or trunc:
            break
    if suffix is not None:
        outpath = save_path / f"simulation_self_{suffix}.gif"
    else:
        outpath = save_path / f"simulation_self.gif"
    imageio.mimwrite(outpath, frames, fps=30)

def win_rate(checkpoint_path, out_dir,which_agent, n_episodes=200):
    env = h_env.HockeyEnv()
    env.one_starts = random.random() > 0.5
    agent = load_agent(checkpoint_path, which_agent, evaluate=True)

    opponent = h_env.BasicOpponent(weak=False)

    wins = 0
    draws = 0

    for ep in range(n_episodes):
        # Alternate sides
        ddpg_is_player_one = (ep % 2 == 0)

        obs_p1, _ = env.reset()
        obs_p2 = env.obs_agent_two()

        done = False
        trunc = False

        for t in range(200):
            if ddpg_is_player_one:
                a1 = agent.act(obs_p1)
                a2 = opponent.act(obs_p2)
            else:
                a1 = opponent.act(obs_p1)
                a2 = agent.act(obs_p2)

            obs_next, _, done, trunc, info_p1 = env.step(np.hstack([a1, a2]))
            info_p2 = env.get_info_agent_two()

            obs_p1 = obs_next
            obs_p2 = env.obs_agent_two()
            if done or trunc:
                break
        # Determine winner
        if ddpg_is_player_one:
            winner = info_p1["winner"]
        else:
            winner = info_p2["winner"]

        if winner == 1:
            wins += 1
        elif winner == 0:
            draws += 1

    winrate = wins / n_episodes
    drawrate = draws / n_episodes

    # Save results
    results = {
        "episodes": n_episodes,
        "wins": wins,
        "draws": draws,
        "winrate": winrate,
        "drawrate": drawrate,
    }

    with open(out_dir / "winrate.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results


def evaluate(which_agent, out_dir, stat_path, checkpoint_path=None):
    print("called evaluate")
    with open(stat_path, 'rb') as f:
        stats = pickle.load(f)

    print(out_dir)
    plo.plot_rewards(stats['rewards'], out_dir)
    plo.plot_lrs(stats['critic_lrs'],"Critic", out_dir)
    plo.plot_lrs(stats['policy_lrs'],"Policy", out_dir)
    plo.plot_losses(stats["c_loss"], "Critic", out_dir)
    plo.plot_losses(stats["p_loss"], "Policy", out_dir)
    plo.plot_losses(stats["a_loss"], "Alpha", out_dir)
    for i in range(5):
        simulate1(checkpoint_path, out_dir,which_agent, suffix=i)
        simulate2(checkpoint_path, out_dir,which_agent, suffix=i)
        simulate_selfplay(checkpoint_path, out_dir,which_agent, suffix=i)
    win_rate(checkpoint_path, out_dir, which_agent)

def evaluate_ddpg(which_agent, out_dir, stat_path, checkpoint_path=None):
    print("called evaluate")
    with open(stat_path, 'rb') as f:
        stats = pickle.load(f)

    print(out_dir)
    plo.plot_rewards(stats['rewards'], out_dir)
    plo.plot_lrs(stats['lrs'],"", out_dir)
    #plo.plot_lrs(stats['policy_lrs'],"Policy", out_dir)
    plo.plot_losses(stats["c_loss"], "Critic", out_dir)
    plo.plot_losses(stats["p_loss"], "Policy", out_dir)
    #plo.plot_losses(stats["p_loss"], "Policy", out_dir)
    #plo.plot_losses(stats["a_loss"], "Alpha", out_dir)
    for i in range(5):
        simulate1(checkpoint_path, out_dir,which_agent, suffix=i)
        simulate2(checkpoint_path, out_dir,which_agent, suffix=i)
        simulate_selfplay(checkpoint_path, out_dir,which_agent, suffix=i)
    win_rate(checkpoint_path, out_dir, which_agent)


if __name__ == '__main__':
    base = Path('checkpoints/2026-02-19_18-49-15-Hockey-SAC/')
    #base=Path('checkpoints/2026-02-17-09:49:10.335538-HockeyEnv-DDPG-eps0.05-l0.0001-y3XYEGI')
    #base= Path('checkpoints/2026-02-16-13:36:33.708753-HockeyEnv-DDPG-eps0.05-l0.0001-8A4nyB8')
    print("hello")
    stat_path = base / 'stats_400.pth'
    checkpoint_path = base / 'checkpoint_400.pth'
    out_dir = base / 'plots'
    #out_dir.mkdir()
    #out_dir.mkdir(parents=True, exist_ok=True)
    #checkpoint_path = Path("results/oAMWN4k-DDPG_HockeyEnv_10000-eps0.1-t32-l0.0001-sNone-Adam-scheduler-False.pth")
    #stat_path = Path("results/oAMWN4k-DDPG_HockeyEnv-eps0.1-t32-l0.0001-sNone-Adam-scheduler-False.pkl")
    #out_dir=Path("plots/re-evaluate")
    evaluate("SAC", out_dir, stat_path, checkpoint_path)

    
    