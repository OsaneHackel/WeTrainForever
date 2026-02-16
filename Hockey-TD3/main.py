import torch 
import numpy as np
import csv
import gymnasium
from gymnasium import spaces
import hockey.hockey_env as hockey_env
from TD3_agent import TD3_Agent
from opponent import make_opponent, get_opponent_action
import os
from commandline_config import build_parser

def train(args):
    # *** Environment setup ***
    env = hockey_env.HockeyEnv()
    full_action_space = env.action_space # actions for player1 || player2
    n_actions_per_player = full_action_space.shape[0] // 2
    agent_action_space = spaces.Box(low=full_action_space.low[:n_actions_per_player],
                                    high=full_action_space.high[:n_actions_per_player],
                                    dtype=full_action_space.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # *** Agent (player 1) ***
    TD3 = TD3_Agent(
        obs_dim = env.observation_space.shape[0],
        act_dim = n_actions_per_player,
        observation_space = env.observation_space,
        action_space = agent_action_space, 
        device = device
        )
    
    if args.resume_from_saved_path: 
        TD3.load(args.resume_from_saved_path)
        print(f"Resume training from {args.resume_from_saved_path}")

    # print arguments?

    # *** Opponent (player 2) ****
    opponent = make_opponent(opponent_type = args.opponent_type, 
                             saved_agent_path = args.saved_agent_path)


    # *** Logging setup ***
    # log file: rewards, losses, win rates
    log_dir = os.path.join(args.output_dir, "logs") # /scratch/<JOB_ID>/hockey_td3/logs/
    os.makedirs(log_dir, exist_ok=True)
    # saved file: save network state
    saved_dir = os.path.join(args.output_dir, "saved")
    os.makedirs(saved_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "training_log.csv")
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "total_steps", "episode_reward", "episode_length",
                         "winner", "critic1_loss", "critic2_loss", "actor_loss"
                         ])

    # *** Training loop ***
    total_env_steps = 0    

    for episode in range(1, args.max_episodes + 1):
        state_agent, info = env.reset() 
        state_opponent = env.obs_agent_two()

        episode_reward = 0.0
        episode_length = 0
        losses = {"critic1_loss": [],
                  "critic2_loss": [],
                  "actor_loss": []
                  }
        ended = False
        while not ended: 
            # Agent's action (player 1, with exploration noise)
            action_1 = TD3.select_action(observation=state_agent,
                                         explore=True)
            # Opponent's action (player 2)
            action_2 = get_opponent_action(opponent=opponent,
                                           opponent_type = args.opponent_type,
                                           agent = TD3,
                                           obs_agent2 = state_opponent)
            # step in the environment
            # observe r and s_next
            (s_next, reward, terminated, truncated, info) = env.step(
                np.hstack([action_1, action_2])
                ) 
            ended = terminated or truncated

            # store transition in replay buffer
            TD3.buffer.add_experience(
                state = state_agent,
                action = action_1,
                reward = reward,
                next_state= s_next,
                ended = ended
            )

            # in self-play, also store opponent's transition
            if args.opponent_type == "current_self":
                s_next_opponent = env.obs_agent_two()
                info_opponent = env.get_info_agent_two()
                reward_opponent = env.get_reward_agent_two(info_opponent)
                TD3.buffer.add_experience(
                    state = state_opponent,
                    action = action_2,
                    reward = reward_opponent,
                    next_state = s_next_opponent,
                    ended = ended
                )

            # *** Train ***
            if total_env_steps >= args.warmup_steps:
                for _ in range(args.train_iter):
                    train_dict = TD3.train_step()
                    if not train_dict.get("skipped"):
                        for key in losses:
                            if key in train_dict:
                                losses[key].append(train_dict[key])

            # advance the state
            state_agent = s_next 
            state_opponent = env.obs_agent_two()
            episode_reward += reward
            episode_length += 1
            total_env_steps += 1


        # *** Log at the end of each episode ***
        winner = info.get("winner", 0)
        mean_c1 = np.mean(losses["critic1_loss"]) if losses["critic1_loss"] else 0
        mean_c2 = np.mean(losses["critic2_loss"]) if losses["critic2_loss"] else 0
        mean_actor = np.mean(losses["actor_loss"]) if losses["actor_loss"] else 0

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_env_steps, episode_reward, episode_length,
                             winner, mean_c1, mean_c2, mean_actor])
            
        if episode % args.save_every == 0:
            path = os.path.join(saved_dir, f"td3_ep{episode}.pt")
            TD3.save(path)
            print(f"Checkpoint saved: {path}")

    # *** Save the final model ***
    final_path = os.path.join(saved_dir, "td3_final.pt")
    TD3.save(final_path)
    print(f"Training  complete. Final model stored in: {final_path}")
    env.close()

def evaluate(args):
    """
    Evaluate a trained agent over many episodes (games)
    """
    env = hockey_env.HockeyEnv(mode=hockey_env.Mode.NORMAL)
    obs_dim = env.observation_space.shape[0]
    full_action_space = env.action_space # actions for player1 || player2
    n_actions_per_player = full_action_space.shape[0] // 2
    agent_action_space = spaces.Box(low=full_action_space.low[:n_actions_per_player],
                                    high=full_action_space.high[:n_actions_per_player],
                                    dtype=full_action_space.dtype)
    TD3 = TD3_Agent(
        obs_dim=obs_dim,
        act_dim=n_actions_per_player,
        observation_space=env.observation_space,
        action_space=agent_action_space,
        device="cpu"
    )
    # player 1 (our agent)
    TD3.load(args.resume_from_saved_path)
    print(f"Load TD3 agent from {args.resume_from_saved_path}")
    # player 2
    opponent = make_opponent(args.opponent_type, 
                             args.saved_agent_path)
    
    n_wins, n_losses, n_ties = 0, 0, 0
    for game in range(1, args.n_games +1):
        state_1, info = env.reset()
        state_2 = env.obs_agent_two()
        episode_reward = 0.0
        ended = False

        while not ended:
            action_1 = TD3.select_action(state_1, explore=False)
            action_2 = get_opponent_action(opponent=opponent, 
                                           opponent_type=args.opponent_type,
                                           agent=TD3,
                                           obs_agent2=state_2)
            (s_next_1, reward, terminated, truncated, info) = env.step(
                np.hstack([action_1, action_2])
            )
            s_next_2 = env.obs_agent_two()
            ended = terminated or truncated
            episode_reward += reward

        winner = info.get("winner", 0)
        if winner == 1:
            n_wins += 1
        elif winner == -1:
            n_losses += 1
        else: 
            n_ties += 1

        if game % 50 == 0:
            print(f"Game {game} / {args.n_games} |"
                  f"Wins {n_wins} Losses {n_losses} Ties {n_ties}")
            
    assert args.n_games == n_wins + n_losses + n_ties, "total number of games doesn't match results"
    print(f"\nResults over {args.n_games} games:")
    print(f"  Wins: {n_wins} ({100 * n_wins/args.n_games:.4f} %)")
    print(f"  Losses: {n_losses} ({100 * n_losses/args.n_games:.4f} %)")
    print(f"  Ties: {n_ties} ({100 * n_ties/args.n_games:.4f} %)")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)

if __name__ == "__main__":
    main()