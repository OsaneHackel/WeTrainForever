import matplotlib.pyplot as plt
import pickle
import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)    

def plot_rewards(rewards, save_path=None, rolling_window=250):
    fig,ax = plt.subplots(figsize=(6,3.5))

    smoothed_rewards = running_mean(rewards, N=rolling_window)
    x = np.arange(rolling_window, rolling_window + len(smoothed_rewards))
    
    ax.plot(x, smoothed_rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    plt.title('Rewards over Episodes')
    print(f"Saving reward plot to {save_path}_rewards.png")
    plt.savefig(save_path / "rewards.png")
    plt.close(fig)

def plot_lrs(lrs, save_path=None):
    fig,ax = plt.subplots(figsize=(6,3.5))

    x = np.arange(len(lrs))
    ax.plot(x, lrs)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Learning Rate')
    plt.title('Learning Rates over Episodes')
    plt.savefig(save_path / "lrs.png")
    plt.close(fig)


