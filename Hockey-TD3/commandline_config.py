import argparse
def build_parser() -> argparse.ArgumentParser:
    # main parser
    parser = argparse.ArgumentParser(description="TD3 Hockey Agent")
    sub_parser = parser.add_subparsers(dest="command", required=True)

    # train subcommands and options
    train_parser = sub_parser.add_parser("train", help="Train the TD3 agent")
    # general training setup
    train_parser.add_argument("--opponent_type", type=str, default="weak",
                              choices=["weak", "strong", "current_self", "pretrained_self"])
    train_parser.add_argument("--saved_agent_path", type=str,
                              default=None)
    train_parser.add_argument("--resume_from_saved_path", type=str, default=None)
    train_parser.add_argument("--save_every", type=int, default=500) # default: 100 saves in between
    train_parser.add_argument("--output_dir", type=str, required=True,
                              help="Directory for logs and checkpoints (/scratch/JOB_ID/)")
    train_parser.add_argument("--max_episodes", type=int, default=50000)
    train_parser.add_argument("--warmup_steps", type=int, default=10000) #FIXME
    train_parser.add_argument("--train_iter", type=int, default=1)#TODO later: 34
    # TD3 hyperparameters
    train_parser.add_argument("--actor_lr", type=float, default=0.0001)
    train_parser.add_argument("--critic_lr", type=float, default=0.0001)
    train_parser.add_argument("--noise_type", type=str, default="OrnsteinU")
    train_parser.add_argument("--batch_size", type=int, default=128)
    train_parser.add_argument("--gamma", type=float, default=0.95)
    train_parser.add_argument("--tau", type=float, default=0.005)
    train_parser.add_argument("--policy_delay", type=int, default=2)
    train_parser.add_argument("--noise_target_policy", type=float, default=0.2)
    train_parser.add_argument("--clip_noise", type=float, default=0.5)
    train_parser.add_argument("--exploration_noise", type=float, default=0.1)
    


    
    # evaluation subcommands
    eval_parser = sub_parser.add_parser("eval", help="Evaluate trained agent",
                                        description="Run evaluation game using a saved checkpoint")
    eval_parser.add_argument("--resume_from_saved_path", type=str, required=True,
                             help="Path to agent checkpoint (.pt) to evaluate")
    eval_parser.add_argument("--opponent_type", type=str, default="weak",
                             choices=["weak", "strong", "current_self", "pretrained_self"],
                             help="Opponent type to evaluate against")
    eval_parser.add_argument("--saved_agent_path", type=str, default=None,
                             help="Required if --opponent_type = pretrained_self. Path to opponent checkpoint (.pt)")
    eval_parser.add_argument("--n_games", type=int, default=200)

    return parser

