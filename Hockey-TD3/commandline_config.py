import argparse
def build_parser() -> argparse.ArgumentParser:
    # main parser
    parser = argparse.ArgumentParser(description="TD3 Hockey Agent")
    sub_parser = parser.add_subparsers(dest="command", required=True)

    # train subcommands and options
    train_parser = sub_parser.add_parser("train", help="Train the TD3 agent")
    train_parser.add_argument("--opponent_type", type=str, default="weak",
                              choices=["weak", "strong", "current_self", "pretrained_self"])
    train_parser.add_argument("--saved_agent_path", type=str,
                              default=None)
    train_parser.add_argument("--max_episodes", type=int, default=50000)
    train_parser.add_argument("--warmup_steps", type=int, default=100) #FIXME
    train_parser.add_argument("--train_iter", type=int, default=34)
    train_parser.add_argument("--resume_from_saved_path", type=str, default=None)
    train_parser.add_argument("--save_every", type=int, default=500) # default: 100 saves in between
    train_parser.add_argument("--output_dir", type=str, required=True,
                              help="Directory for logs and checkpoints (/scratch/JOB_ID/)")
    
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

