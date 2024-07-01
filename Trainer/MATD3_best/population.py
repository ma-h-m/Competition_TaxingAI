import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import os,sys
from matd3 import MATD3
sys.path.append(os.path.abspath('../..'))
from experience_replay import replay_buffer
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import random
import time

torch.autograd.set_detect_anomaly(True)

'''
matd3
'''
class matd3_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        self.eval_env = copy.copy(envs)
        # todo add
        self.noise = args.noise_std_init
        self.epsilon = args.epsilon
        self.args.n_agents = self.envs.households.n_households + 1

        # # start to build the network.
        self.args.gov_obs_dim = self.envs.government.observation_space.shape[0]
        self.args.gov_action_dim = self.envs.government.action_space.shape[0]
        self.args.house_obs_dim = self.envs.households.observation_space.shape[0]
        self.args.house_action_dim = self.envs.households.action_space.shape[1]
        self.args.prev = 4
        self.max_files = args.pop_size
        self.opp_times = 5
        self.cat_gov_state = np.zeros((self.args.prev, self.args.gov_obs_dim))
        self.cat_house_state = np.zeros((self.args.prev, self.envs.households.n_households, (self.args.house_obs_dim - self.args.gov_obs_dim)))

        # define the replay buffer
        self.agents = self._init_agents()
        self.buffer = replay_buffer(self.args.buffer_size)
        self.writer = SummaryWriter(log_dir='../runs/matd3_population/'+args.role)
        
    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = MATD3(self.args, i)
            agents.append(agent)
        return agents
    
    def observation_wrapper(self, global_obs, private_obs):
        # global
        global_obs[0] /= 1e7
        global_obs[1] /= 1e6
        global_obs[3] /= 1e5
        global_obs[4] /= 1e5
        global_obs[-1] /= 1e5
        private_obs[:, 1] /= 1e5
        return global_obs, private_obs

    def learn(self):
        epoch = 0
        while True:
            if self.args.role == 'gov':
                dir_path = './model/gov/'
                eval_agents = os.listdir(dir_path)
            else:
                dir_path = './model/household/'
                eval_agents = os.listdir(dir_path)
            if len(eval_agents) <= self.max_files:
                time.sleep(60)
                continue
            epoch += 1
            scores = [0.0 for _ in range(len(eval_agents))]
            gov_scores = [0.0 for _ in range(len(eval_agents))]
            house_scores = [0.0 for _ in range(len(eval_agents))]
            curr_idx = 0
            for e_agent_name in eval_agents:
                house_id = int(random.choice([0,1,2,3]))
                other_house_ids = [0,1,2,3]
                other_house_ids.remove(house_id)
                if self.args.role == 'gov':
                    eval_agent_path = './model/gov/' + e_agent_name
                    self.agents[4].actor_network.load_state_dict(torch.load(eval_agent_path, map_location="cuda"))
                else:
                    eval_agent_path = './model/household/' + e_agent_name
                    self.agents[house_id].actor_network.load_state_dict(torch.load(eval_agent_path, map_location="cuda"))
                # load others
                gov_rew = []
                house_rew = []
                for t_ in range(self.opp_times): # random choose 30 times to estimate
                    if self.args.role == 'gov': # load others
                        dir_path = './model/household/'
                        filenames = os.listdir(dir_path)
                        for idx in range(4):
                            try:
                                self.agents[idx].actor_network.load_state_dict(torch.load("./model/household/"+random.choice(filenames), map_location="cuda"))
                            except:
                                pass
                    else:
                        dir_path = './model/household/'
                        filenames = os.listdir(dir_path)
                        for idx in other_house_ids:
                            try:
                                self.agents[idx].actor_network.load_state_dict(torch.load("./model/household/"+random.choice(filenames), map_location="cuda"))
                            except:
                                pass
                        dir_path = './model/gov/'
                        filenames = os.listdir(dir_path)
                        try:
                            self.agents[4].actor_network.load_state_dict(torch.load("./model/gov/"+random.choice(filenames), map_location="cuda"))
                        except:
                            pass
                
                    for agent in self.agents:
                        agent.eval_model()
                    # start to do the evaluation
                    mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
                    # store rewards and step
                    gov_rew.append(mean_gov_rewards)
                    house_rew.append(mean_house_rewards)
                gov_scores[curr_idx] = np.mean(gov_rew)
                house_scores[curr_idx] = np.mean(house_rew)
                scores[curr_idx] = np.mean(gov_rew) if self.args.role == 'gov' else np.mean(house_rew)
                curr_idx += 1
                print(
                    '[{}] Epoch: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, years:{:.3f}'.format(
                        datetime.now(), epoch, np.mean(gov_rew), np.mean(house_rew),years))
            
            self.writer.add_scalar('house_reward_mean', np.mean(house_scores), global_step=epoch)
            self.writer.add_scalar('gov_reward', np.mean(gov_scores), global_step=epoch)
            self.writer.add_scalar('house_reward_min', np.min(house_scores), global_step=epoch)
            self.writer.add_scalar('gov_reward_min', np.min(gov_scores), global_step=epoch)
            self.writer.add_scalar('house_reward_max', np.max(house_scores), global_step=epoch)
            self.writer.add_scalar('gov_reward_max', np.max(gov_scores), global_step=epoch)
            last_idx = np.argsort(scores)
            del_num =  len(scores) - self.max_files
            del_files = ['./model/'+self.args.role+'/'+eval_agents[_idx] for _idx in last_idx[:del_num]]
            for del_file in del_files:
                try:  
                    os.remove(del_file)  
                    print(f"Deleted file: {del_file}")  
                except Exception as e:  
                    print(f"Error deleting file {del_file}: {e}") 
                
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor

    def _evaluate_agent(self):
        total_gov_reward = 0
        total_house_reward = 0
        total_steps = 0
        mean_tax = 0
        mean_wealth = 0
        mean_post_income = 0
        gdp = 0
        income_gini = 0
        wealth_gini = 0
        for epoch_i in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
            cat_gov_state = np.zeros((self.args.prev, self.args.gov_obs_dim))
            cat_house_state = np.zeros((self.args.prev, self.envs.households.n_households, (self.args.house_obs_dim - self.args.gov_obs_dim)))
            cat_gov_state[:-1] = cat_gov_state[1:]
            cat_gov_state[-1] = deepcopy(global_obs)
            cat_house_state[:-1] = cat_house_state[1:]
            cat_house_state[-1] = deepcopy(private_obs)
            flat_global_obs, flat_private_obs = cat_gov_state.reshape(-1), cat_house_state.transpose(1,0,2).reshape((self.eval_env.households.n_households, -1))
            episode_gov_reward = 0
            episode_mean_house_reward = 0
            step_count = 0
            episode_mean_tax = []
            episode_mean_wealth = []
            episode_mean_post_income = []
            episode_gdp = []
            episode_income_gini = []
            episode_wealth_gini = []
        
            while True:
                with torch.no_grad():
                    action = self._evaluate_get_action(flat_global_obs, flat_private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
                    cat_gov_state[:-1] = cat_gov_state[1:]
                    cat_gov_state[-1] = deepcopy(next_global_obs)
                    cat_house_state[:-1] = cat_house_state[1:]
                    cat_house_state[-1] = deepcopy(next_private_obs)
                    flat_next_global_obs, flat_next_private_obs = cat_gov_state.reshape(-1), cat_house_state.transpose(1,0,2).reshape((self.envs.households.n_households, -1))
                step_count += 1
                episode_gov_reward += gov_reward
                episode_mean_house_reward += np.mean(house_reward)
                episode_mean_tax.append(np.mean(self.eval_env.tax_array))
                episode_mean_wealth.append(np.mean(self.eval_env.households.at_next))
                episode_mean_post_income.append(np.mean(self.eval_env.post_income))
                episode_gdp.append(self.eval_env.per_household_gdp)
                episode_income_gini.append(self.eval_env.income_gini)
                episode_wealth_gini.append(self.eval_env.wealth_gini)
                if done:
                    break
                flat_global_obs, flat_private_obs = flat_next_global_obs, flat_next_private_obs
            total_gov_reward += episode_gov_reward
            total_house_reward += episode_mean_house_reward
            total_steps += step_count
            mean_tax += np.mean(episode_mean_tax)
            mean_wealth += np.mean(episode_mean_wealth)
            mean_post_income += np.mean(episode_mean_post_income)
            gdp += np.mean(episode_gdp)
            income_gini += np.mean(episode_income_gini)
            wealth_gini += np.mean(episode_wealth_gini)
    
        avg_gov_reward = total_gov_reward / self.args.eval_episodes
        avg_house_reward = total_house_reward / self.args.eval_episodes
        mean_step = total_steps / self.args.eval_episodes
        avg_mean_tax = mean_tax / self.args.eval_episodes
        avg_mean_wealth = mean_wealth / self.args.eval_episodes
        avg_mean_post_income = mean_post_income / self.args.eval_episodes
        avg_gdp = gdp / self.args.eval_episodes
        avg_income_gini = income_gini / self.args.eval_episodes
        avg_wealth_gini = wealth_gini / self.args.eval_episodes
        return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, \
               avg_wealth_gini, mean_step

    def _evaluate_get_action(self, global_obs, private_obs):
        global_obs_tensor = self._get_tensor_inputs(global_obs)
        private_obs_tensor = self._get_tensor_inputs(private_obs)
        hou_action = np.zeros((self.envs.households.n_households, self.args.house_action_dim))
        for agent_id, agent in enumerate(self.agents):
            if agent_id == self.args.n_agents - 1:  # government agent
                gov_action = agent.select_action(global_obs_tensor, 0, 0, agent_id)
            else:  # households agent
                obs = torch.cat([global_obs_tensor, private_obs_tensor[agent_id]], dim=-1)
                hou_action[agent_id] = agent.select_action(obs, 0, 0, agent_id)
    
        action = {self.envs.government.name: gov_action,
                  self.envs.households.name: hou_action}
        return action
