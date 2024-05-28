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
import shutil
from scipy.special import softmax

import importlib.util
import sys
import tempfile
def dynamic_import_class(module_name, file_path, class_name):
    # 获取文件所在目录的父目录的父目录，即顶级包目录
    module_dir = os.path.dirname(file_path)
    import sys    
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
    # top_level_package_dir = os.path.abspath(os.path.join(module_dir, os.pardir, os.pardir, os.pardir))
    top_level_package_dir = "/home/mhm/workspace/Competition_TaxingAI/Server/policy_pools/test1/long_term/models/independent_ppo-1716880432-0/utils"
    # 将顶级包目录添加到 sys.path
    sys.path.insert(0, top_level_package_dir)
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取指定类
    cls = getattr(module, class_name)
    
    # 移除顶级包目录
    sys.path.pop(0)
    
    return cls

def _get_code_path(model_path):
    return os.path.join(os.path.dirname(os.path.dirname(model_path)), "agent.py")
# 现版本household和gov的模型是耦合的，没办法单独加载一个类
# TODO:分离household和gov的模型
def load_model(model_path, env, cfg):
    path = os.path.dirname(os.path.dirname(model_path))
    Agent = dynamic_import_class("agent", _get_code_path(model_path), "agent")
    agent = Agent(env, cfg, os.join(path, "run","house_net.pt"), os.join(path, "run","gov_net.pt"))
    return agent

def select_policy(policy_pool: pd.DataFrame, env, cfg, temperature=1.0):

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
    agent = load_model(selected_policy_path, env, cfg)
    return agent

def evaluate_policy_pools(cfg_path = "default"):
    yaml_cfg = OmegaConf.load(f'./Server/cfg/{cfg_path}.yaml')
    set_seeds(yaml_cfg.seed, cuda = yaml_cfg.Trainer.cuda)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    env = economic_society(yaml_cfg.Environment)
    government_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_government.csv")
    household_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_household.csv")
    # select_policy(government_policy_pool, env, yaml_cfg.Trainer)
    agent = load_model(government_policy_pool.iloc[0]['path'], env, yaml_cfg.Trainer)

evaluate_policy_pools()
