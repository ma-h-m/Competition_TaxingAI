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
    last_dir = os.path.basename(module_dir)
    top_level_package_dir = os.path.abspath(os.path.join(module_dir, os.pardir, os.pardir, os.pardir))
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    parts = module_dir.split('/')
    parts = parts[1:]
    tmp_path = '.'.join(parts)
    # 设置父包，以支持相对导入
    # setattr(module, '__package__', 'policy_pools.test1.long_term.models.' + last_dir)
    setattr(module, '__package__', tmp_path)
    
    # 将顶级包目录添加到 sys.path
    sys.path.insert(0, top_level_package_dir)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取指定类
    cls = getattr(module, class_name)
    
    # 移除顶级包目录
    sys.path.pop(0)
    
    return cls
def _get_code_path(model_path):
    return os.path.join(os.path.dirname(os.path.dirname(model_path)), "agent.py")

def load_model(model_path, env, cfg): #agent_type: "households" or "government"
    if model_path.endswith("house_net.pt"):
        agent_type = "households"
    elif model_path.endswith("gov_net.pt"):
        agent_type = "government"
    else:
        raise ValueError("Invalid model path")
    
    path = os.path.dirname(os.path.dirname(model_path))
    Agent = dynamic_import_class("agent", _get_code_path(model_path), "agent")
    if agent_type == "households":
        agent = Agent(env, cfg, house_net_path=model_path, test=True)
    else:
        agent = Agent(env, cfg, gov_net_path=model_path, test=True)

    # agent.load_pretrained_weights(model_path, agent_type)
 

    # agent = Agent(env, cfg, os.path.join(path, "run","house_net.pt"), os.path.join(path, "run","gov_net.pt"))
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
    # selected_policy = policy_pool.loc[selected_index]
    selected_index = 0
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
    # government_agent = select_policy(government_policy_pool, env, yaml_cfg.Trainer)

    government_agent_num = 0
    households_agent_num = 0
    for entity in yaml_cfg.Environment.Entities:
        if entity.entity_name == "government":
            government_agent_num = entity.entity_args.n
        if entity.entity_name == "household":
            households_agent_num = entity.entity_args.n
    government_agents = []
    for i in range(government_agent_num):
        government_agents.append(select_policy(government_policy_pool, env, yaml_cfg.Trainer))
    household_agents = []
    for i in range(households_agent_num):
        household_agents.append(select_policy(household_policy_pool, env, yaml_cfg.Trainer))
    for i in range(20):
            # 重置环境
            global_obs, private_obs = env.reset()
            
            done = False
            step_count = 0
            while not done:
                # 获取政府行为
                government_actions = []
                
                government_actions = government_agents[0].get_one_action(global_obs, private_obs, isHousehold=False)
                
                
                # 获取家庭行为
                household_actions = []
                for agent, private_ob in zip(household_agents, private_obs):
                # for agent in household_agents:
                    household_action = agent.get_one_action(global_obs, private_ob, isHousehold=True)
                    household_actions.append(household_action)
                
                # 构造动作字典
                action_dict = {
                    "government": np.array(government_actions),  
                    "Household": np.array(household_actions)
                }
                
                # 执行一步
                next_global_state, next_private_state, government_reward, households_reward, done = env.step(action_dict)
                
                # 更新观察
                global_obs = next_global_state
                private_obs = next_private_state
                
                # 打印步数和奖励
                step_count += 1
                print(f"Step {step_count}: Government Reward = {government_reward}, Households Reward = {np.mean(households_reward)}")

    return 
    

evaluate_policy_pools('n4')
