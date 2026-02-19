import optparse

def build_parser():
    parser = optparse.OptionParser()
    parser.add_option('-e', '--env', action='store', type='string',
                 dest='env_name', default='Pendulum-v1',
                 help='Gymnasium environment (default %default)')
    parser.add_option('--noise_type', action='store', type='string',
                 dest='noise_type', default='Gaussian',
                 help='Exploration noise: Gaussian | OrnsteinU | Pink (default %default)')
    parser.add_option('--noise_beta', action='store', type='float',
                 dest='noise_beta', default=1.0,
                 help='Beta for colored/pink noise (default %default)')
    parser.add_option('--exploration_noise', action='store', type='float',
                 dest='exploration_noise', default=0.1,
                 help='Exploration noise scale (default %default)')
    parser.add_option('-m', '--maxepisodes', action='store', type='int',
                 dest='max_episodes', default=5000,
                 help='Number of training episodes (default %default)')
    parser.add_option('--warmup_steps', action='store', type='int',
                 dest='warmup_steps', default=1000,
                 help='Steps before training starts (default %default)')
    parser.add_option('--train_iter', action='store', type='int',
                 dest='train_iter', default=32,
                 help='Gradient updates per episode (default %default)')
    parser.add_option('--batch_size', action='store', type='int',
                 dest='batch_size', default=128)
    parser.add_option('--actor_lr', action='store', type='float',
                 dest='actor_lr', default=1e-3)
    parser.add_option('--critic_lr', action='store', type='float',
                 dest='critic_lr', default=1e-3)
    parser.add_option('--gamma', action='store', type='float',
                 dest='gamma', default=0.99)
    parser.add_option('--tau', action='store', type='float',
                 dest='tau', default=0.005)
    parser.add_option('--policy_delay', action='store', type='int',
                 dest='policy_delay', default=2)
    parser.add_option('--log_interval', action='store', type='int',
                 dest='log_interval', default=20,
                 help='Print avg reward every N episodes (default %default)')
    parser.add_option('-s', '--seed', action='store', type='int',
                 dest='seed', default=None)
    return parser
