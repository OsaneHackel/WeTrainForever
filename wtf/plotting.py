import matplotlib.pyplot as plt
import pickle
import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)    

def plot_rewards(rewards, save_path=None):
    smoothed_rewards = running_mean(rewards, N=100)
    x = np.arange(250, 250 + len(smoothed_rewards))
    plt.plot(x, smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    print(f"Saving reward plot to {save_path}_rewards.png")
    plt.savefig(f"{save_path}_rewards.png")

def plot_lrs(lrs, save_path=None):
    x = np.arange(len(lrs))
    plt.plot(x, lrs)
    plt.xlabel('Episode')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rates over Episodes')
    plt.savefig((f"{save_path}_lrs.png"))


