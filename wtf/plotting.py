import matplotlib.pyplot as plt
import pickle
import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)    

def plot_rewards(rewards):
    smoothed_rewards = running_mean(rewards, N=100)
    x = np.arange(250, 250 + len(smoothed_rewards))
    plt.plot(x, smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.savefig('plots/rewards.png')

def plot_lrs(lrs, save_path=None):
    x = np.arange(len(lrs))
    plt.plot(x, lrs)
    plt.xlabel('Episode')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rates over Episodes')
    plt.savefig(save_path)


def plot():
    file_path = 'results/DDPG_HockeyEnv-eps0.1-t32-l0.0001-sNone-stat-qMKRTiw.pkl'
    with open(file_path, 'rb') as f:
        stats = pickle.load(f)
    rewards = stats['rewards']
    run_id = file_path.split('-stat-')[-1].split('.pkl')[0]
    plot_rewards(rewards, save_path=f'plots/{run_id}_rewards.png')
    plot_lrs(stats['lrs'], save_path=f'plots/{run_id}_lrs.png')

if __name__ == '__main__':
    plot()
