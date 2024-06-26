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
import copy
import importlib.util
import sys
import tempfile
import random
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

def load_full_model(model_path, env,cfg):
    path = os.path.dirname(os.path.dirname(model_path))
    Agent = dynamic_import_class("agent", _get_code_path(model_path), "agent")
    agent = Agent(copy.deepcopy(env), cfg, house_net_path=os.path.join(path, "run","house_net.pt"), gov_net_path=os.path.join(path, "run","gov_net.pt"), test=True)
    return agent

def select_policy(policy_pool: pd.DataFrame, env, cfg, temperature=100.0, sigma = 1.0):

    # 获取score列
    scores = policy_pool['score'].values
    evaluated_times = policy_pool['evaluated_times'].values

    # 归一化score列
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # 对归一化的score应用softmax函数
    probabilities = softmax(normalized_scores / temperature)

    # 使用 UCB 方法计算每个策略的选择权重
    total_evaluations = np.sum(evaluated_times)
    exploration_term = np.sqrt((2 * np.log(total_evaluations + 1)) / (evaluated_times + 1))

    # 调整概率：结合 UCB 的思想
    adjusted_probabilities = probabilities + exploration_term * sigma
    
    adjusted_probabilities /= np.sum(adjusted_probabilities)  # 归一化以确保是概率
    


    # print("Normalized Scores:", normalized_scores)
    # print("Probabilities:", probabilities)

    # 根据概率分布进行抽样
    np.random.seed()
    selected_index = np.random.choice(policy_pool.index, p=probabilities)
    
    # selected_index = 0 # for debug
    # print(f"Selected index: {selected_index}")
    selected_policy = policy_pool.loc[selected_index]
    selected_policy_path = selected_policy['path']
    
    agent = load_model(selected_policy_path, env, cfg)
    return agent
def update_log(model_path, role, score, policy_pool): #role: "household" or "government"
    log_path = f"{ROOT_DIR}/log_{role}.csv"
    # 找到 path 列等于 model_path 的行
    mask = policy_pool['path'] == model_path
    
    # 确保至少有一个匹配项
    if not mask.any():
        raise ValueError(f"No entry with path '{model_path}' found in policy_pool")

    # 更新 score 列和 evaluated_times 列
    policy_pool.loc[mask, 'score'] = (policy_pool.loc[mask, 'score'] * policy_pool.loc[mask, 'evaluated_times'] + score) / (policy_pool.loc[mask, 'evaluated_times'] + 1)
    policy_pool.loc[mask, 'evaluated_times'] += 1
    policy_pool.to_csv(log_path, index=False)
        
def evaluate_one_model_in_policy_pools(cfg_path = "default", selected_index = 0):
    yaml_cfg = OmegaConf.load(f'./Server/cfg/{cfg_path}.yaml')
    set_seeds(yaml_cfg.seed, cuda = yaml_cfg.Trainer.cuda)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    env = economic_society(yaml_cfg.Environment)
    government_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_government.csv")
    # household_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_household.csv")
    # government_agent = select_policy(government_policy_pool, env, yaml_cfg.Trainer)

    selected_policy = government_policy_pool.loc[selected_index]
    selected_policy_path = selected_policy['path']
    agent = load_full_model(selected_policy_path, env, yaml_cfg.Trainer)
    
    agent.test()
    
# def tmp_debug_func(full_model_agent, government_agents, household_agents, env):
#     random.seed(0)
#     torch.manual_seed(0)
#     global_obs, private_obs = env.reset()

#     # gov_reward = 0
#     # mean_house_reward = 0
#     # step_count = 0
#     # mean_tax = []
#     # mean_wealth = []
#     # mean_post_income = []
#     # gdp = []
#     # income_gini = []
#     # wealth_gini = []
#     # house_reward = [0 for _ in range(len(household_agents))]
#     # done = False

#     episode_gov_reward = 0
#     episode_mean_house_reward = 0
#     step_count = 0
#     episode_mean_tax = []
#     episode_mean_wealth = []
#     episode_mean_post_income = []
#     episode_gdp = []
#     episode_income_gini = []
#     episode_wealth_gini = []
#     households_agent_num = len(household_agents)
#     episode_house_reward = [0 for _ in range(households_agent_num)]
#     done = False
#     # full_model_agent.env = copy.deepcopy(env)
#     while not done:

#     #     with torch.no_grad():
#     #         tmp_global_obs = copy.deepcopy(global_obs)
#     #         tmp_private_obs = copy.deepcopy(private_obs)
#     #         tmp_global_obs, tmp_private_obs = full_model_agent.observation_wrapper(tmp_global_obs, tmp_private_obs)
#     #         # action = full_model_agent._evaluate_get_action(tmp_global_obs, tmp_private_obs)
#     #         pis = full_model_agent._evaluate_get_pis(tmp_global_obs, tmp_private_obs)
#     #         # random.seed(0)
#     #         tmp_action = full_model_agent._evaluate_get_action(tmp_global_obs, tmp_private_obs)



#         # random.seed(0)
#         government_actions = government_agents[0].get_one_action(global_obs, private_obs, isHousehold=False)
#         # Get household actions
#         # household_actions = [
#         #     agent.get_one_action(global_obs, private_ob, isHousehold=True) 
#         #     for agent, private_ob in zip(household_agents, private_obs)
#         # ]
#         household_pis = [
#             agent.get_one_pis(global_obs, private_ob, isHousehold=True)
#             for agent, private_ob in zip(household_agents, private_obs)
#         ]
#         # household_actions =household_agents[0].select_actions(household_pis)
#         tmp_means = [item[0] for item in household_pis]
#         tmp_stds = [item[1] for item in household_pis]

#         # 将列表转化为单个张量
#         mean_tensor = torch.cat(tmp_means).unsqueeze(0)
#         std_tensor = torch.cat(tmp_stds).unsqueeze(0)
#         household_actions = household_agents[0].select_actions((mean_tensor, std_tensor))
        
#         # Construct action dictionary
#         action_dict = {
#             "government": np.array(government_actions),  
#             "Household": np.array(household_actions)
#         }
        
#         # Perform one step in the environment
#         next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action_dict)
#         # Update observations
#         global_obs = next_global_obs
#         private_obs = next_private_obs
        
#         # Accumulate rewards and metrics
#         step_count += 1
#         episode_gov_reward += gov_reward
#         episode_mean_house_reward += np.mean(house_reward)
#         episode_mean_tax.append(np.mean(env.tax_array))
#         episode_mean_wealth.append(np.mean(env.households.at_next))
#         episode_mean_post_income.append(np.mean(env.post_income))
#         episode_gdp.append(env.per_household_gdp)
#         episode_income_gini.append(env.income_gini)
#         episode_wealth_gini.append(env.wealth_gini)
#     print(f"episode_gov_reward: {episode_gov_reward}, episode_mean_house_reward: {episode_mean_house_reward}, step_count: {step_count}, episode_mean_tax: {np.mean(episode_mean_tax)}, episode_mean_wealth: {np.mean(episode_mean_wealth)}, episode_mean_post_income: {np.mean(episode_mean_post_income)}, episode_gdp: {np.mean(episode_gdp)}, episode_income_gini: {np.mean(episode_income_gini)}, episode_wealth_gini: {np.mean(episode_wealth_gini)}")

#     episode_gov_reward = 0
#     episode_mean_house_reward = 0
#     step_count = 0
#     episode_mean_tax = []
#     episode_mean_wealth = []
#     episode_mean_post_income = []
#     episode_gdp = []
#     episode_income_gini = []
#     episode_wealth_gini = []
#     done = False
#     global_obs, private_obs = full_model_agent.observation_wrapper(global_obs, private_obs)

#     while True:
#         with torch.no_grad():
#             action = full_model_agent._evaluate_get_action(global_obs, private_obs)
#             next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action) #full_model_agent.eval_env.step(action)
#             # another_next_global_obs, another_next_private_obs, another_gov_reward, another_house_reward, another_done = env.step(action)
#             # another_another_next_global_obs, another_another_next_private_obs, another_another_gov_reward, another_another_house_reward, another_another_done = full_model_agent.envs.step(action)
#             next_global_obs, next_private_obs = full_model_agent.observation_wrapper(next_global_obs, next_private_obs)

#         step_count += 1
#         # Accumulate rewards and metrics
#         step_count += 1
#         episode_gov_reward += gov_reward
#         episode_mean_house_reward += np.mean(house_reward)
#         episode_mean_tax.append(np.mean(env.tax_array))
#         episode_mean_wealth.append(np.mean(env.households.at_next))
#         episode_mean_post_income.append(np.mean(env.post_income))
#         episode_gdp.append(env.per_household_gdp)
#         episode_income_gini.append(env.income_gini)
#         episode_wealth_gini.append(env.wealth_gini)
#         if done:
#             break

#         global_obs = next_global_obs
#         private_obs = next_private_obs
#     print(f"episode_gov_reward: {episode_gov_reward}, episode_mean_house_reward: {episode_mean_house_reward}, step_count: {step_count}, episode_mean_tax: {np.mean(episode_mean_tax)}, episode_mean_wealth: {np.mean(episode_mean_wealth)}, episode_mean_post_income: {np.mean(episode_mean_post_income)}, episode_gdp: {np.mean(episode_gdp)}, episode_income_gini: {np.mean(episode_income_gini)}, episode_wealth_gini: {np.mean(episode_wealth_gini)}")



def evaluate_policy_pools(cfg_path = "n4", lock = None, temperature = 100.0):
    yaml_cfg = OmegaConf.load(f'./Server/cfg/{cfg_path}.yaml')
    set_seeds(yaml_cfg.seed, cuda = yaml_cfg.Trainer.cuda)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    env = economic_society(yaml_cfg.Environment)
    if lock:
        lock.acquire()
    try:
        government_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_government.csv")
        household_policy_pool = pd.read_csv(f"{ROOT_DIR}/log_household.csv")
    finally:
        if lock:
            lock.release()
    # government_agent = select_policy(government_policy_pool, env, yaml_cfg.Trainer)

    government_agent_num = 0
    households_agent_num = 0
    for entity in yaml_cfg.Environment.Entities:
        if entity.entity_name == "government":
            government_agent_num = entity.entity_args.n
        if entity.entity_name == "household":
            households_agent_num = entity.entity_args.n
    
    government_agents = [select_policy(government_policy_pool, env, yaml_cfg.Trainer, temperature) for _ in range(government_agent_num)]
    household_agents = [select_policy(household_policy_pool, env, yaml_cfg.Trainer, temperature) for _ in range(households_agent_num)]
    

    # env_for_full_model = economic_society(yaml_cfg.Environment)

    # full_model_agent = load_full_model(government_policy_pool.loc[0]['path'], env_for_full_model, yaml_cfg.Trainer)


    

    total_gov_reward = 0
    # total_house_reward = 0
    total_steps = 0
    mean_tax = 0
    mean_wealth = 0
    mean_post_income = 0
    gdp = 0
    income_gini = 0
    wealth_gini = 0

    eval_episodes = 3
    total_house_reward = [0 for _ in range(households_agent_num)]



    # tmp_debug_func(full_model_agent, government_agents, household_agents, env)
    # return

    for episode_i in range(eval_episodes):
        # random.seed(0)
        # torch.manual_seed(0)
        global_obs, private_obs = env.reset()
        episode_gov_reward = 0
        episode_mean_house_reward = 0
        step_count = 0
        episode_mean_tax = []
        episode_mean_wealth = []
        episode_mean_post_income = []
        episode_gdp = []
        episode_income_gini = []
        episode_wealth_gini = []
        episode_house_reward = [0 for _ in range(households_agent_num)]
        done = False
        while not done:
            # Debugging with full model agent
            # Get government actions
            government_actions = government_agents[0].get_one_action(global_obs, private_obs, isHousehold=False)
            
            household_actions = []
            for agent, private_ob in zip(household_agents, private_obs):
                household_actions.append(agent.get_one_action(global_obs, private_ob, isHousehold=True))
            # Get household actions
            # household_pis = [
            #     agent.get_one_pis(global_obs, private_ob, isHousehold=True)
            #     for agent, private_ob in zip(household_agents, private_obs)
            # ]
            # # household_actions =household_agents[0].select_actions(household_pis)
            # tmp_means = [item[0] for item in household_pis]
            # tmp_stds = [item[1] for item in household_pis]

            # # 将列表转化为单个张量
            # mean_tensor = torch.cat(tmp_means).unsqueeze(0)
            # std_tensor = torch.cat(tmp_stds).unsqueeze(0)
            # # household_actions = household_agents[0].select_actions((mean_tensor, std_tensor))
            # household_actions = []
            # for i in range(households_agent_num):
            #     tmp_household_actions = household_agents[i].select_actions((mean_tensor, std_tensor))
            #     household_actions.append(tmp_household_actions[i])
            # for agent in household_agents:
            #     tmp_household_actions = agent.select_actions((mean_tensor, std_tensor))
            #     household_actions.append(tmp_household_actions)
            




            # Construct action dictionary
            action_dict = {
                "government": np.array(government_actions),  
                "Household": np.array(household_actions)
            }
            
            # Perform one step in the environment
            next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action_dict)
            # Update observations
            global_obs = next_global_obs
            private_obs = next_private_obs
            
            # Accumulate rewards and metrics
            step_count += 1
            episode_gov_reward += gov_reward
            episode_mean_house_reward += np.mean(house_reward)
            episode_mean_tax.append(np.mean(env.tax_array))
            episode_mean_wealth.append(np.mean(env.households.at_next))
            episode_mean_post_income.append(np.mean(env.post_income))
            episode_gdp.append(env.per_household_gdp)
            episode_income_gini.append(env.income_gini)
            episode_wealth_gini.append(env.wealth_gini)
            for i in range(households_agent_num):
                episode_house_reward[i] += house_reward[i]

        # Accumulate episode results
        total_gov_reward += episode_gov_reward
        # total_house_reward += episode_mean_house_reward
        for i in range(households_agent_num):
            total_house_reward[i] += episode_house_reward[i]
        total_steps += step_count
        mean_tax += np.mean(episode_mean_tax)
        mean_wealth += np.mean(episode_mean_wealth)
        mean_post_income += np.mean(episode_mean_post_income)
        gdp += np.mean(episode_gdp)
        income_gini += np.mean(episode_income_gini)
        wealth_gini += np.mean(episode_wealth_gini)
        
    # Calculate average results across all episodes
    avg_gov_reward = total_gov_reward / eval_episodes
    # avg_house_reward = total_house_reward / eval_episodes
    mean_step = total_steps / eval_episodes
    avg_mean_tax = mean_tax / eval_episodes
    avg_mean_wealth = mean_wealth / eval_episodes
    avg_mean_post_income = mean_post_income / eval_episodes
    avg_gdp = gdp / eval_episodes
    avg_income_gini = income_gini / eval_episodes
    avg_wealth_gini = wealth_gini / eval_episodes
    avg_house_reward = [reward / eval_episodes for reward in total_house_reward]
    if lock:
        lock.acquire()
    try:
        update_log(government_agents[0].gov_net_path, "government", avg_gov_reward, government_policy_pool)
        for i in range(households_agent_num):
            update_log(household_agents[i].house_net_path, "household", avg_house_reward[i], household_policy_pool)
    finally:
        if lock:
            lock.release()

    return {
        "avg_gov_reward": avg_gov_reward,
        "avg_house_reward": avg_house_reward,
        "avg_mean_tax": avg_mean_tax,
        "avg_mean_wealth": avg_mean_wealth,
        "avg_mean_post_income": avg_mean_post_income,
        "avg_gdp": avg_gdp,
        "avg_income_gini": avg_income_gini,
        "avg_wealth_gini": avg_wealth_gini,
        "mean_step": mean_step,
        "avg_house_reward": avg_house_reward
    }
# for _ in range(10):
#     result = evaluate_policy_pools('n4')
#     print(result)