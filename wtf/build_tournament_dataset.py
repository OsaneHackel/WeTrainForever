'''
This scipt was generated with AI support
This script processes tournament replay files to build a dataset of transitions.
Each replay file contains a sequence of transitions from a single game, and I
extract states, actions, rewards, next states, and done flags to create a structured dataset.'''

import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hockey.hockey_env as h_env

INPUT_DIR = Path("tournament_replay")
OUTPUT_FILE = Path("tournament_dataset.pkl")
VERIFY_NEXT_STATE = False   # set True for debugging
PRINT_STATS = True

def load_game(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_reward(info):
    reward = 25.0 * info["winner"]
    reward += 2.0 * info["reward_closeness_to_puck"]
    reward += 3.0 * info["reward_touch_puck"]
    return reward

def compute_reward(env, state, action):
    """
    Computes reward by forcing environment into given state
    and stepping once with zero-opponent policy.
    """
    env._state = state.copy()

    # zero opponent action
    opponent_action = np.zeros_like(action)
    full_action = np.hstack([action, opponent_action])

    next_state, reward, done, trunc, info = env.step(full_action)
    reward = get_reward(info)

    return reward, next_state, done or trunc, info

def build_dataset():
    env = h_env.HockeyEnv()

    dataset = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
        "info":[],
    }

    files = sorted(INPUT_DIR.glob("*.pkl"))
    print(f"\nFound {len(files)} replay files")

    total_transitions = 0
    skipped_games = 0

    for file in tqdm(files):
        game = load_game(file)
        transitions = game.get("transitions", [])

        if len(transitions) == 0:
            skipped_games += 1
            continue

        for tr in transitions:
            state = np.array(tr["state"], dtype=np.float32)
            action = np.array(tr["action"], dtype=np.float32)
            next_state_logged = np.array(tr["next_state"], dtype=np.float32)
            done = tr["done"]

            # Recompute reward
            reward, next_state_env, done_env, info = compute_reward(env, state, action)

            if VERIFY_NEXT_STATE:
                diff = np.linalg.norm(next_state_env - next_state_logged)
                if diff > 1e-3:
                    print(f"Warning: next_state mismatch (||diff||={diff:.4f})")

            dataset["states"].append(state)
            dataset["actions"].append(action)
            dataset["rewards"].append(reward)
            dataset["next_states"].append(next_state_logged)
            dataset["dones"].append(done)
            dataset["info"].append(info)

            total_transitions += 1
    print("\nFinished rebuilding dataset")
    print(f"Total transitions: {total_transitions}")
    print(f"Skipped empty games: {skipped_games}")

    with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(dataset, f)

    print(f"\nSaved dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()