# I asked ChatGPT to come up with some of these argument names and descriptions

import argparse
import numpy as np
def build_parser() -> argparse.ArgumentParser:
    # main parser
    parser = argparse.ArgumentParser(description="TD3 Hockey Agent")
    sub_parser = parser.add_subparsers(dest="command", required=True)

    # train subcommands and options
    train_parser = sub_parser.add_parser("train", help="Train the TD3 agent")
    # general training setup
    train_parser.add_argument("--opponent_type", type=str, default="weak",
                              choices=["weak", "strong", "current_self", "pretrained_self", 
                                       "pool_basic_and_frozen_self", "sac",
                                       "pool_with_sac", "pool_and_self_play"])
    train_parser.add_argument("--opponent_odds", type=str, 
                              default=None,
                              help="JSON string of opponentnt weights, e.g. '{\"weak\":1,\"strong\":1,\"frozen_agent\":3}'")
    # '{"weak": 0.35, "strong": 0.45, "frozen_agent": 0.2}'
    train_parser.add_argument("--saved_agent_path", type=str,
                              default=None,
                              help="Required if --opponent_type=pretrained_self (.pt)")
    train_parser.add_argument("--resume_from_saved_path", type=str, default=None,
                              help="Path to the agent checkpoint from where to start training on (.pt)")
    train_parser.add_argument("--use_saved_hyperparams", action="store_true",
                              help="If set, use hyperparameters from the saved checkpoint instead of CLI args")
    train_parser.add_argument("--save_every", type=int, default=500) # default: 100 saves in between
    train_parser.add_argument("--output_dir", type=str, required=True,
                              help="Directory for logs and checkpoints (/scratch/JOB_ID/)")
    train_parser.add_argument("--max_episodes", type=int, default=50000)
    train_parser.add_argument("--warmup_steps", type=int, default=10_000) #FIXME
    train_parser.add_argument("--train_iter", type=int, default=4) #TODO later: 34
    train_parser.add_argument("--validate_every", type=int, default=200)
    train_parser.add_argument("--n_val_games", type=int, default=300)

    train_parser.add_argument("--sac_path", type=str, default=None,
                              help="Path to Osane's saved SAC agent (.pth)"
                                   "Required for --opponent_type sac or pool_with_sac")
    train_parser.add_argument("--sac_folder_path", type=str, default=None)

    train_parser.add_argument("--alternate_sides", action="store_true",
                          help="If set, alternate training between player 1 and player 2 each episode")
    train_parser.add_argument("--p2_probability", type=float, default=0.5,
                            help="Probability of playing as player 2 each episode (used with --alternate_sides)")
    
    # train using data from games (comprl server)
    train_parser.add_argument("--prefill_from", type=str, default=None,
                              help="Path to directory of tournament files (.pkl) to prefill the replay buffer")
    train_parser.add_argument("--prefill_wins_only", action="store_true",
                              help="If set, only prefill with data from won games")
    
    train_parser.add_argument("--seed", type=int, default=42)
    # TD3 hyperparameters
    train_parser.add_argument("--actor_lr", type=float, default=0.0001)
    train_parser.add_argument("--critic_lr", type=float, default=0.0001)
    train_parser.add_argument("--noise_type", type=str, default="Pink",
                              choices=["Gaussian", "OrnsteinU", "Pink"])
    train_parser.add_argument("--noise_beta", type=float, default=1.0,
                              help="Parameter for colored noise (0=White Gaussian Noise, 1=Pink, 2=Red)")
    train_parser.add_argument("--batch_size", type=int, default=128)
    train_parser.add_argument("--gamma", type=float, default=0.95)
    train_parser.add_argument("--tau", type=float, default=0.005)
    train_parser.add_argument("--policy_delay", type=int, default=3)
    train_parser.add_argument("--noise_target_policy", type=float, default=0.2)
    train_parser.add_argument("--clip_noise", type=float, default=0.5)
    train_parser.add_argument("--exploration_noise", type=float, default=0.1)
    # Prioritized Experience Replay hyperparams
    train_parser.add_argument("--use_PrioritizedExpReplay", action="store_true",
                              help="If set, use Prioritized Experience Replay Buffer instead of uniform Buffer")
    train_parser.add_argument("--PER_alpha", type=float, default=0.6)
    train_parser.add_argument("--PER_beta_init", type=float, default=0.4)
    train_parser.add_argument("--PER_beta_n_steps", type=int, default=100_000,
                              help="Number of steps over which beta annearly from PER_beta_init to 1.0")


    
    # evaluation subcommands
    eval_parser = sub_parser.add_parser("eval", help="Evaluate trained agent",
                                        description="Run evaluation game using a saved checkpoint")
    eval_parser.add_argument("--resume_from_saved_path", type=str, required=True,
                             help="Path to agent checkpoint (.pt) to evaluate")
    eval_parser.add_argument("--opponent_type", type=str, default="weak",
                             choices=["weak", "strong", "current_self", "pretrained_self", "sac"],
                             help="Opponent type to evaluate against")
    eval_parser.add_argument("--saved_agent_path", type=str, default=None,
                             help="Required if --opponent_type = pretrained_self. Path to opponent checkpoint (.pt)")
    eval_parser.add_argument("--n_games", type=int, default=200)
    eval_parser.add_argument("--play_as_player2", action="store_true",
                             help="If set, our agent plays as player 2 and the opponent as player 1")
    eval_parser.add_argument("--sac_path", type=str, default=None,
                             help="Required if --opponent_type = sac, path to SAC checkpoint (.pth)")
    eval_parser.add_argument("--sac_folder_path", type=str, default=None,
                             help="Path to the sibling repo root containing wtf/agents/SAC.py")

    return parser

