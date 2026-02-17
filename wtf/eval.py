import numpy as np
import pickle
import torch
import imageio
from pathlib import Path

import hockey.hockey_env as h_env

import wtf.plotting as plo
from wtf.agents.DDPG import DDPGAgent
from wtf.utils import load_ddpg

#TODO: use eps=0.0 for the evaluation

def simulate1(checkpoint_path, save_path):
    env = h_env.HockeyEnv()
    ddpg = load_ddpg(checkpoint_path)
    opponent = h_env.BasicOpponent(weak=False)
    obs_opponent, _ = env.reset()
    obs_ddpg = env.obs_agent_two()
    frames = []
    for t in range(200):
        a_op = opponent.act(obs_opponent)
        a_ddpg =ddpg.act(obs_ddpg)
        obs_next, r, done, trunc, _ = env.step(np.hstack([a_ddpg, a_op]))
        frames.append(env.render(mode='rgb_array'))
        obs_opponent = obs_next
        obs_ddpg = env.obs_agent_two()
        if done or trunc:
            break
    
    imageio.mimwrite(save_path / "simulation_blue.gif", frames, fps=30)

def simulate2(checkpoint_path, save_path):
    env = h_env.HockeyEnv()
    ddpg = load_ddpg(checkpoint_path)
    opponent = h_env.BasicOpponent(weak=False)
    obs_ddpg, _ = env.reset()
    obs_opponent = env.obs_agent_two()
    frames = []
    for t in range(200):
        a_ddpg =ddpg.act(obs_ddpg)
        a_op = opponent.act(obs_opponent)
        obs_next, r, done, trunc, _ = env.step(np.hstack([a_op, a_ddpg]))
        frames.append(env.render(mode='rgb_array'))
        obs_ddpg = obs_next
        obs_opponent = env.obs_agent_two()
        if done or trunc:
            break
    
    imageio.mimwrite(save_path / "simulation_red.gif", frames, fps=30)

def win_rate(checkpoint_path, out_dir, n_episodes=100):
    env = h_env.HockeyEnv()
    ddpg = load_ddpg(checkpoint_path)
    ddpg.eps = 0.0  # disable exploration during eval

    opponent = h_env.BasicOpponent(weak=True)

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
                a1 = ddpg.act(obs_p1)
                a2 = opponent.act(obs_p2)
            else:
                a1 = opponent.act(obs_p1)
                a2 = ddpg.act(obs_p2)

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


def evaluate(out_dir, stat_path, checkpoint_path=None):
    with open(stat_path, 'rb') as f:
        stats = pickle.load(f)
        #stats=torch.load(f)
    #print(stats)
    print(out_dir)
    plo.plot_rewards(stats['rewards'], out_dir)
    plo.plot_lrs(stats['lrs'], out_dir)
    simulate1(checkpoint_path, out_dir)
    simulate2(checkpoint_path, out_dir)
    win_rate(checkpoint_path, out_dir)

if __name__ == '__main__':
    #base = Path('checkpoints/2026-02-16-16:33:09.644209-HockeyEnv-DDPG-eps0.05-l0.0001-dddwHCs/')
    #base=Path('checkpoints/2026-02-17-09:49:10.335538-HockeyEnv-DDPG-eps0.05-l0.0001-y3XYEGI')
    base= Path('checkpoints/2026-02-16-13:36:33.708753-HockeyEnv-DDPG-eps0.05-l0.0001-8A4nyB8')
    stat_path = base / 'stats_3500-t32.pth'
    checkpoint_path = base / 'checkpoint_3500-t32.pth'
    out_dir = base / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate(out_dir, stat_path, checkpoint_path)

    
    