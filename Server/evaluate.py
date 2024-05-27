import numpy as np

from env.env_core import economic_society


from utils.seeds import set_seeds
from arguments import get_args
import os
import torch
import yaml
import argparse
from omegaconf import OmegaConf
import pandas as pd
ROOT_DIR = "Server/policy_pools"

from scipy.special import softmax

import importlib.util
import sys

def dynamic_import_class(module_name, file_path, class_name):
    # 动态加载模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取指定类
    return getattr(module, class_name)

def select_policy(policy_pool: pd.DataFrame, temperature=1.0):

    # 获取score列
    scores = policy_pool['score'].values

    # 归一化score列
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # 对归一化的score应用softmax函数
    probabilities = softmax(normalized_scores / temperature)

    # 根据概率分布进行抽样
    selected_index = np.random.choice(policy_pool.index, p=probabilities)
    selected_policy = policy_pool.loc[selected_index]
    selected_policy_path = selected_policy['path']
    
    return selected_policy

def evaluate_policy_pools(cfg_path = "default"):
    yaml_cfg = OmegaConf.load(f'./Server/cfg/{cfg_path}.yaml')
    set_seeds(yaml_cfg.seed, cuda = yaml_cfg.Trainer.cuda)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    env = economic_society(yaml_cfg.Environment)
    government_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_government.csv")
    household_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_household.csv")
    select_policy(government_policy_pool)

evaluate_policy_pools()
