import numpy as np

from env.env_core import economic_society
# from agents.baseline import agent
# from agents.rule_based import rule_agent
# from agents.independent_ppo import ppo_agent
# from agents.calibration import calibration_agent
# from agents.BMFAC import BMFAC_agent
# from agents.MADDPG.MAAC import maddpg_agent
# from agents.MADDPG_block.MAAC import maddpg_agent
from utils.seeds import set_seeds
from arguments import get_args
import os
import torch
import yaml
import argparse
from omegaconf import OmegaConf

# from agents.model_self.agent import agent
from client import *

import shutil


from dynamic_import import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument("--alg", type=str, default='ac', help="ac, rule_based, independent")
    parser.add_argument("--task", type=str, default='gdp', help="gini, social_welfare, gdp_gini")
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    parser.add_argument('--n_households', type=int, default=100, help='the number of total households')
    parser.add_argument('--seed', type=int, default=1, help='the random seed')
    parser.add_argument('--hidden_size', type=int, default=128, help='[64, 128, 256]')
    parser.add_argument('--q_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--p_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--batch_size', type=int, default=64, help='[32, 64, 128, 256]')
    parser.add_argument('--update_cycles', type=int, default=100, help='[10，100，1000]')
    parser.add_argument('--update_freq', type=int, default=10, help='[10，20，30]')
    parser.add_argument('--initial_train', type=int, default=10, help='[10，100，200]')
    
    # parser.add_argument('--update_freq', type=int, default=1, help='the number of total households')
    # parser.add_argument('--initial_train', type=int, default=2000, help='the number of total households')


    parser.add_argument('--long_term_policy_update_freq', type=int, default=10, help='interval of updating long term policy pool')
    # parser.add_argument("--long_term_policy_pool_size", type=int, default=30000, help="size of long term policy pool")
    # parser.add_argument("--short_term_policy_pool_size", type=int, default=10, help="size of short term policy pool")
    # parser.add_argument("--top_k_policy_update_freq", type=int, default=10, help="interval of updating top k policy pool")
    # parser.add_argument("--top_k_policy_pool_size", type=int, default=10, help="size of top k policy pool")
    # parser.add_argument("--freq_of_pushing_moodels_to_server", type=int, default=10, help="frequency of pushing models to server")
    parser.add_argument("--push_new_policy_to_server", type=int, default=1, help="whether to push new policy to server")
    parser.add_argument("--upload_freq", type=int, default=10, help="frequency of uploading models to server")
    parser.add_argument("--local_epoch", type=int, default=50, help="local epoch")
    args = parser.parse_args()
    return args



def tuning(cfg):
    # 要修改的参数
    # maddpg
    hidden_size_list = [64, 128, 256]
    lr_list = [3e-3, 3e-4, 3e-5]
    batch_size_list = [32, 64, 128, 256]
    for hidden_i in hidden_size_list:
        for lr_i in lr_list:
            for batch_i in batch_size_list:
                cfg.hidden_size=hidden_i
                cfg.q_lr = lr_i
                cfg.p_lr = lr_i
                cfg.batch_size = batch_i

    return cfg

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    path = args.config
    # yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
    current_path = os.path.join(os.getcwd(), "TaxAI_3rd_version")

    yaml_cfg = OmegaConf.load(f'{current_path}/agents/cfg/n4.yaml')
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
    
    # yaml_cfg.Trainer["long_term_policy_update_freq"] = args.long_term_policy_update_freq
    # yaml_cfg.Trainer["long_term_policy_pool_size"] = args.long_term_policy_pool_size
    # yaml_cfg.Trainer["short_term_policy_pool_size"] = args.short_term_policy_pool_size
    # yaml_cfg.Trainer["freq_of_pushing_moodels_to_server"] = args.freq_of_pushing_moodels_to_server
    yaml_cfg.Trainer["upload_freq"] = args.upload_freq
    yaml_cfg.Trainer["local_training_epoch"] = args.local_epoch
    # yaml_cfg.Trainer["top_k_policy_update_freq"] = args.top_k_policy_update_freq
    # yaml_cfg.Trainer["top_k_policy_pool_size"] = args.top_k_policy_pool_size

    
    set_seeds(yaml_cfg.seed, cuda=yaml_cfg.Trainer["cuda"])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    env = economic_society(yaml_cfg.Environment)

    model_pool_path = "agents/model_pool"


    # if args.alg == "ac":
    #     trainer = agent(env, yaml_cfg.Trainer)
    # elif args.alg == "rule_based":
    #     trainer = rule_agent(env, yaml_cfg.Trainer)
    # elif args.alg == "ppo":
    #     trainer = ppo_agent(env, yaml_cfg.Trainer)
    # elif args.alg == "bmfac":
    #     trainer = BMFAC_agent(env, yaml_cfg.Trainer)
    # elif args.alg == "maddpg":
    #     trainer = maddpg_agent(env, yaml_cfg.Trainer)
    # else:
    #     trainer = calibration_agent(env, yaml_cfg.Trainer)
    # start to learn
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    
    initial_communicate_with_server(USER_ID)
    code_path = "TaxAI_3rd_version/agents/model_self"
    external_policy_path = "TaxAI_3rd_version/agents/model_pools/models_from_server"

    while True:


        if not args.push_new_policy_to_server: # If not push new policy to server, fetch a top_k model from server to agents/model_self
            shutil.rmtree(code_path)
            fetch_random_top_k_model(user_id=USER_ID, dest_dir= code_path)
            print("Fetch model from server to agents/model_self")
        
        fetch_random_models(user_id=USER_ID, dest_dir=external_policy_path)
        Agent = dynamic_import_class("agent", os.path.join(code_path, 'agent.py'), "agent")
        trainer = Agent(env, yaml_cfg.Trainer)


        trainer.path_dict = {"code_path": code_path, "external_policy_path": external_policy_path}
        print("Successfully load model. Start to train.")
        trainer.learn()


    # trainer = agent(env, yaml_cfg.Trainer)

    # trainer.learn()
    # trainer.test()
    # # close the environment
    env.close()


