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
    agent = Agent(env, cfg, house_net_path=os.path.join(path, "run","house_net.pt"), gov_net_path=os.path.join(path, "run","gov_net.pt"), test=True)
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
    
    selected_index = 0 # for debug

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
    
    government_agents = [select_policy(government_policy_pool, env, yaml_cfg.Trainer) for _ in range(government_agent_num)]
    household_agents = [select_policy(household_policy_pool, env, yaml_cfg.Trainer) for _ in range(households_agent_num)]
    
    # government_agents = []
    # for i in range(government_agent_num):
    #     government_agents.append(select_policy(government_policy_pool, env, yaml_cfg.Trainer))
    # household_agents = []
    # for i in range(households_agent_num):
    #     household_agents.append(select_policy(household_policy_pool, env, yaml_cfg.Trainer))


    full_model_agent = load_full_model(government_policy_pool.loc[0]['path'], env, yaml_cfg.Trainer)
    mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = full_model_agent._evaluate_agent()
    print(f"mean_gov_rewards: {mean_gov_rewards}, mean_house_rewards: {mean_house_rewards}, avg_mean_tax: {avg_mean_tax}, avg_mean_wealth: {avg_mean_wealth}, avg_mean_post_income: {avg_mean_post_income}, avg_gdp: {avg_gdp}, avg_income_gini: {avg_income_gini}, avg_wealth_gini: {avg_wealth_gini}, years: {years}")
    # full_model_agent.test()


    total_gov_reward = 0
    # total_house_reward = 0
    total_steps = 0
    mean_tax = 0
    mean_wealth = 0
    mean_post_income = 0
    gdp = 0
    income_gini = 0
    wealth_gini = 0

    eval_episodes = 100
    total_house_reward = [0 for _ in range(households_agent_num)]
    for episode_i in range(eval_episodes):
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
            # with torch.no_grad():
            #     tmp_global_obs = copy.deepcopy(global_obs)
            #     tmp_private_obs = copy.deepcopy(private_obs)
            #     tmp_global_obs, tmp_private_obs = full_model_agent.observation_wrapper(tmp_global_obs, tmp_private_obs)
            #     # action = full_model_agent._evaluate_get_action(tmp_global_obs, tmp_private_obs)
            #     pis = full_model_agent._evaluate_get_pis(tmp_global_obs, tmp_private_obs)

            # reference_household_pis = torch.stack(pis[1], dim=0)
            # government_pis = government_agents[0].get_one_pis(global_obs, private_obs, isHousehold=False)
            # household_pis = [
            #     agent.get_one_pis(global_obs, private_ob, isHousehold=True) 
            #     for agent, private_ob in zip(household_agents, private_obs)
            # ]
            # # 将 pis 转换成与 government_pis 和 household_pis 相同的数据结构
            # government_pis_from_pis = pis[0]

            # # 由于 pis[1][0] 和 pis[1][1] 是三维张量，我们需要将其转换为与 household_pis 匹配的形式
            # household_pis_from_pis = []
            # for i in range(pis[1][0].size(1)):
            #     household_pis_from_pis.append((
            #         pis[1][0][:, i, :],
            #         pis[1][1][:, i, :]
            #     ))

            # # 检查 government_pis 是否相等
            # government_equal = all(torch.equal(a, b) for a, b in zip(government_pis, government_pis_from_pis))

            # # 检查 household_pis 是否相等
            # household_equal = all(
            #     all(torch.allclose(a, b, atol=1e-5)for a, b in zip(hp, hp_from_pis))
            #     for hp, hp_from_pis in zip(household_pis, household_pis_from_pis)
            # )
            # if not government_equal:
            #     print("Government PIS are not equal.")
            # if not household_equal:
            #     print("Household PIS are not equal.")

            # Get government actions
            government_actions = government_agents[0].get_one_action(global_obs, private_obs, isHousehold=False)
            


            
            # Get household actions
            household_actions = [
                agent.get_one_action(global_obs, private_ob, isHousehold=True) 
                for agent, private_ob in zip(household_agents, private_obs)
            ]
            
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
    update_log(government_agents[0].gov_net_path, "government", avg_gov_reward, government_policy_pool)
    for i in range(households_agent_num):
        update_log(household_agents[i].house_net_path, "household", avg_house_reward[i], household_policy_pool)
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



    # for i in range(20):
    #         # 重置环境
    #         global_obs, private_obs = env.reset()
            
    #         done = False
    #         step_count = 0
    #         while not done:
    #             # 获取政府行为
    #             government_actions = []
                
    #             government_actions = government_agents[0].get_one_action(global_obs, private_obs, isHousehold=False)
                
                
    #             # 获取家庭行为
    #             household_actions = []
    #             for agent, private_ob in zip(household_agents, private_obs):
    #             # for agent in household_agents:
    #                 household_action = agent.get_one_action(global_obs, private_ob, isHousehold=True)
    #                 household_actions.append(household_action)
                
    #             # 构造动作字典
    #             action_dict = {
    #                 "government": np.array(government_actions),  
    #                 "Household": np.array(household_actions)
    #             }
                
    #             # 执行一步
    #             next_global_state, next_private_state, government_reward, households_reward, done = env.step(action_dict)
                
    #             # 更新观察
    #             global_obs = next_global_state
    #             private_obs = next_private_state
                
    #             # 打印步数和奖励
    #             step_count += 1
    #             print(f"Step {step_count}: Government Reward = {government_reward}, Households Reward = {np.mean(households_reward)}")

    # return 
# evaluate_one_model_in_policy_pools('n4', 0)    

result = evaluate_policy_pools('n4')
print(result)