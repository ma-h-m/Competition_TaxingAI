import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from TaxAI.env.env_core import economic_society
from population import matd3_agent
import os
import torch
import yaml
import argparse
from omegaconf import OmegaConf

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, default='gov', help="gov, household")
    parser.add_argument("--task", type=str, default='gdp_gini', help="gini, social_welfare, gdp_gini")
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    parser.add_argument('--n_households', type=int, default=100, help='the number of total households')
    parser.add_argument('--seed', type=int, default=1, help='the random seed')
    parser.add_argument('--load_new', type=int, default=5, help='load new others')
    parser.add_argument('--pop_size', type=int, default=100, help='load new others')
    parser.add_argument('--hidden_size', type=int, default=128, help='[64, 128, 256]')
    parser.add_argument('--q_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--p_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--batch_size', type=int, default=64, help='[32, 64, 128, 256]')
    parser.add_argument('--update_cycles', type=int, default=100, help='[10，100，1000]')
    parser.add_argument('--update_freq', type=int, default=10, help='[10，20，30]')
    parser.add_argument('--initial_train', type=int, default=10, help='[10，100，200]')
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    yaml_path = '../TaxAI/cfg/n4.yaml'
    yaml_cfg = OmegaConf.load(yaml_path)
    # todo if local run code
    # yaml_cfg = OmegaConf.load(f'D:\\code\\AI-TaxingPolicy\\AI-TaxingPolicy\\cfg\\default.yaml')
    yaml_cfg.Trainer["n_households"] = args.n_households
    yaml_cfg.Environment.Entities[1]["entity_args"].n = args.n_households
    yaml_cfg.Environment.env_core["env_args"].gov_task = args.task
    yaml_cfg.seed = args.seed
    
    '''tuning'''
    # tuning(yaml_cfg)
    yaml_cfg.Trainer["hidden_size"] = args.hidden_size
    yaml_cfg.Trainer["q_lr"] = args.q_lr
    yaml_cfg.Trainer["p_lr"] = args.p_lr
    yaml_cfg.Trainer["batch_size"] = args.batch_size
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    env = economic_society(yaml_cfg.Environment)
    args.epsilon = yaml_cfg.Trainer['epsilon']
    args.cuda = yaml_cfg.Trainer['cuda']
    args.buffer_size = yaml_cfg.Trainer['buffer_size']
    args.n_epochs = yaml_cfg.Trainer['n_epochs']
    args.epoch_length = yaml_cfg.Trainer['epoch_length']
    args.gamma = yaml_cfg.Trainer['gamma']
    args.tau = yaml_cfg.Trainer['tau']
    args.display_interval = yaml_cfg.Trainer['display_interval']
    args.eval_episodes = 2
    args.save_interval = 50
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    print(args.role)
    trainer = matd3_agent(env, args)
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    trainer.learn()


