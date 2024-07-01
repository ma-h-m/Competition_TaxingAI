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

torch.autograd.set_detect_anomaly(True)

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

'''
matd3
'''
class agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        self.eval_env = copy.copy(envs)
        # todo add
        self.noise = args.noise_std_init
        self.epsilon = args.epsilon
        self.pop_id = args.pop_id
        self.args.n_agents = self.envs.households.n_households + 1

        # # start to build the network.
        self.args.gov_obs_dim = self.envs.government.observation_space.shape[0]
        self.args.gov_action_dim = self.envs.government.action_space.shape[0]
        self.args.house_obs_dim = self.envs.households.observation_space.shape[0]
        self.args.house_action_dim = self.envs.households.action_space.shape[1]
        self.args.prev = 4
        self.cat_gov_state = np.zeros((self.args.prev, self.args.gov_obs_dim+1))
        self.cat_house_state = np.zeros((self.args.prev, self.envs.households.n_households, (self.args.house_obs_dim - self.args.gov_obs_dim)))

        # define the replay buffer
        self.agents = self._init_agents()
        self.buffer = replay_buffer(self.args.buffer_size)
        self.writer = SummaryWriter(log_dir='../runs/matd3_EA/'+args.role+'/'+str(self.pop_id))
        
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
        # reset the environment
        year_obs = np.zeros((1,))
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        self.cat_gov_state[:-1] = self.cat_gov_state[1:]
        self.cat_gov_state[-1] = deepcopy(np.concatenate([global_obs, year_obs/self.args.epoch_length], -1))
        self.cat_house_state[:-1] = self.cat_house_state[1:]
        self.cat_house_state[-1] = deepcopy(private_obs)
        flat_global_obs, flat_private_obs = self.cat_gov_state.reshape(-1), self.cat_house_state.transpose(1,0,2).reshape((self.envs.households.n_households, -1))
        gov_rew = []
        house_rew = []
        epochs = []
        sum_actor_loss = 0
        sum_critic_loss = 0
        houses = self.agents[:4]
        random.shuffle(houses)
        self.agents[:4] = houses
        try:
            if self.args.role == 'gov':
                dir_path = './model_pop/gov/'
                filenames = os.listdir(dir_path)
                self.agents[4].actor_network.load_state_dict(torch.load("./model/gov/"+random.choice(filenames), map_location="cuda"))
            else:
                dir_path = './model_pop/household/'
                filenames = os.listdir(dir_path)
                self.agents[0].actor_network.load_state_dict(torch.load("./model/household/"+random.choice(filenames), map_location="cuda"))
                self.agents[1].actor_network.load_state_dict(torch.load("./model/household/"+random.choice(filenames), map_location="cuda"))
                self.agents[2].actor_network.load_state_dict(torch.load("./model/household/"+random.choice(filenames), map_location="cuda"))
                self.agents[3].actor_network.load_state_dict(torch.load("./model/household/"+random.choice(filenames), map_location="cuda"))
        except:
            pass
        for epoch in range(self.args.n_epochs):
            houses = self.agents[:4]
            random.shuffle(houses)
            self.agents[:4] = houses
            if epoch % self.args.load_new == 0:
                load_cnt = 0
                while load_cnt<=3:
                    load_cnt += 1
                    try:
                        if self.args.role == 'gov':
                            for _id in range(4):
                                if random.random()<0.4:
                                    dir_path = './model/household/'
                                    filenames = os.listdir(dir_path)
                                else:
                                    dir_path = './model_pop/household/'
                                    filenames = os.listdir(dir_path)
                                self.agents[_id].actor_network.load_state_dict(torch.load(dir_path+random.choice(filenames), map_location="cuda"))
                        else:
                            if random.random()<0.4:
                                dir_path = './model/gov/'
                                filenames = os.listdir(dir_path)
                            else:
                                dir_path = './model_pop/gov/'
                                filenames = os.listdir(dir_path)
                            filenames = os.listdir(dir_path)
                            self.agents[4].actor_network.load_state_dict(torch.load(dir_path+random.choice(filenames), map_location="cuda"))
                        print('load new others-------------------------')
                        break
                    except:
                        pass
            # for each epoch, it will reset the environment
            for t in range(self.args.epoch_length):
                '''
                for each agent, get its action from observation
                '''
                mem_flat_global_obs = deepcopy(flat_global_obs)
                mem_flat_private_obs = deepcopy(flat_private_obs)
                global_obs_tensor = self._get_tensor_inputs(flat_global_obs)
                private_obs_tensor = self._get_tensor_inputs(flat_private_obs)
                hou_action = np.zeros((self.envs.households.n_households, self.args.house_action_dim))
                for agent_id, agent in enumerate(self.agents):
                    if agent_id == self.args.n_agents-1:  # government agent
                        gov_action = agent.select_action(global_obs_tensor, self.noise, self.epsilon, agent_id)
                    else:   # households agent
                        obs = torch.cat([global_obs_tensor, private_obs_tensor[agent_id]], dim=-1)
                        hou_action[agent_id] = agent.select_action(obs, self.noise, self.epsilon, agent_id)

                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: hou_action}
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                gov_reward = 1 / (1 + np.exp(-gov_reward))
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
                year_obs[0] += 1.0
                self.cat_gov_state[:-1] = self.cat_gov_state[1:]
                self.cat_gov_state[-1] = deepcopy(np.concatenate([next_global_obs, year_obs/self.args.epoch_length], -1))
                self.cat_house_state[:-1] = self.cat_house_state[1:]
                self.cat_house_state[-1] = deepcopy(next_private_obs)
                flat_next_global_obs, flat_next_private_obs = self.cat_gov_state.reshape(-1), self.cat_house_state.transpose(1,0,2).reshape((self.envs.households.n_households, -1))
                mem_flat_next_global_obs = deepcopy(flat_next_global_obs)
                mem_flat_next_private_obs = deepcopy(flat_next_private_obs)
                # store the episodes
                gov_reward += 0.01
                house_reward /= 10.0
                house_reward += 0.5
                self.buffer.add(mem_flat_global_obs, mem_flat_private_obs, gov_action, hou_action, gov_reward, house_reward,
                                mem_flat_next_global_obs, mem_flat_next_private_obs, float(done))
                # past_mean_house_action = self.multiple_households_mean_action(hou_action)
                flat_global_obs = flat_next_global_obs
                flat_private_obs = flat_next_private_obs
                if done:
                    year_obs = np.zeros((1,))
                    # if done, reset the environment
                    global_obs, private_obs = self.envs.reset()
                    global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
                    self.cat_gov_state = np.zeros((self.args.prev, self.args.gov_obs_dim+1))
                    self.cat_house_state = np.zeros((self.args.prev, self.envs.households.n_households, (self.args.house_obs_dim - self.args.gov_obs_dim)))
                    self.cat_gov_state[:-1] = self.cat_gov_state[1:]
                    self.cat_gov_state[-1] = deepcopy(np.concatenate([global_obs, year_obs/self.args.epoch_length], -1))
                    self.cat_house_state[:-1] = self.cat_house_state[1:]
                    self.cat_house_state[-1] = deepcopy(private_obs)
                    flat_global_obs, flat_private_obs = self.cat_gov_state.reshape(-1), self.cat_house_state.transpose(1,0,2).reshape((self.envs.households.n_households, -1))
                # for _ in range(self.args.update_cycles):
                uodate_freq = 50    # 更新频率
                if t % uodate_freq == 0:
                    # after collect the samples, start to update the network
                    transitions = self.buffer.sample(self.args.batch_size)
                    sum_actor_loss = 0
                    sum_critic_loss = 0
                    if self.args.role == 'gov':
                        other_agents = self.agents.copy()
                        other_agents.remove(self.agents[4])
                        actor_loss, critic_loss = self.agents[4].train(transitions, other_agents, 4)
                        sum_actor_loss += actor_loss
                        sum_critic_loss += critic_loss
                    else:
                        curr_idx = 0
                        for agent in self.agents[:4]:
                            other_agents = self.agents.copy()
                            other_agents.remove(agent)
                            actor_loss, critic_loss = agent.train(transitions, other_agents, curr_idx)
                            sum_actor_loss += actor_loss
                            sum_critic_loss += critic_loss
                            curr_idx += 1
            # print the log information
            if epoch % self.args.display_interval == 0:
                for agent in self.agents:
                    agent.eval_model()
                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
                # store rewards and step
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                epochs.append(now_step)
                self.writer.add_scalar('house_reward_mean', mean_house_rewards, global_step=now_step)
                self.writer.add_scalar('gov_reward', mean_gov_rewards, global_step=now_step)
                self.writer.add_scalar('years', years, global_step=now_step)
                self.writer.add_scalar('wealth gini', avg_wealth_gini, global_step=now_step)
                self.writer.add_scalar('income gini', avg_income_gini, global_step=now_step)
                self.writer.add_scalar('GDP', avg_gdp, global_step=now_step)
                self.writer.add_scalar('tax per households', avg_mean_tax, global_step=now_step)
                self.writer.add_scalar('post income per households', avg_mean_post_income, global_step=now_step)
                self.writer.add_scalar('wealth per households', avg_mean_wealth, global_step=now_step)
                self.writer.add_scalar('actor loss', sum_actor_loss, global_step=now_step)
                self.writer.add_scalar('critic loss', sum_critic_loss, global_step=now_step)
                self.writer.add_scalar('noise', self.noise, global_step=now_step)
                self.writer.add_scalar('eps', self.epsilon, global_step=now_step)

                print(
                    '[{}] Epoch: {} / {}, Frames: {}, gov_Rewards: {:.3f}, house_Rewards: {:.3f}, years:{:.3f}, actor_loss: {:.3f}, critic_loss: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_gov_rewards, mean_house_rewards,years, sum_actor_loss, sum_critic_loss))
                for agent in self.agents:
                    agent.train_model()
                # save models
            if epoch % self.args.save_interval == 0:
                print('save:'+self.args.role)
                if self.args.role == 'gov':
                    torch.save(self.agents[4].actor_network.state_dict(), 'model/'+self.args.role+'/agent_'+str(self.pop_id)+'_'+str(epoch)+'.pt')
                    torch.save(self.agents[4].actor_network.state_dict(), 'model_pop/'+self.args.role+'/agent_'+str(self.pop_id)+'_'+str(epoch)+'.pt')
                    dir_path = './model_pop/gov/'
                    filenames = os.listdir(dir_path)
                    self.agents[4].actor_network.load_state_dict(torch.load(dir_path+random.choice(filenames), map_location="cuda"))
                else:
                    torch.save(self.agents[0].actor_network.state_dict(), 'model/'+self.args.role+'/agent_'+str(self.pop_id)+'_1_'+str(epoch)+'.pt')
                    torch.save(self.agents[1].actor_network.state_dict(), 'model/'+self.args.role+'/agent_'+str(self.pop_id)+'_2_'+str(epoch)+'.pt')
                    torch.save(self.agents[2].actor_network.state_dict(), 'model/'+self.args.role+'/agent_'+str(self.pop_id)+'_3_'+str(epoch)+'.pt')
                    torch.save(self.agents[3].actor_network.state_dict(), 'model/'+self.args.role+'/agent_'+str(self.pop_id)+'_4_'+str(epoch)+'.pt')
                    
                    torch.save(self.agents[0].actor_network.state_dict(), 'model_pop/'+self.args.role+'/agent_'+str(self.pop_id)+'_1_'+str(epoch)+'.pt')
                    torch.save(self.agents[1].actor_network.state_dict(), 'model_pop/'+self.args.role+'/agent_'+str(self.pop_id)+'_2_'+str(epoch)+'.pt')
                    torch.save(self.agents[2].actor_network.state_dict(), 'model_pop/'+self.args.role+'/agent_'+str(self.pop_id)+'_3_'+str(epoch)+'.pt')
                    torch.save(self.agents[3].actor_network.state_dict(), 'model_pop/'+self.args.role+'/agent_'+str(self.pop_id)+'_4_'+str(epoch)+'.pt')
                    dir_path = './model_pop/household/'
                    filenames = os.listdir(dir_path)
                    self.agents[0].actor_network.load_state_dict(torch.load(dir_path+random.choice(filenames), map_location="cuda"))
                    self.agents[1].actor_network.load_state_dict(torch.load(dir_path+random.choice(filenames), map_location="cuda"))
                    self.agents[2].actor_network.load_state_dict(torch.load(dir_path+random.choice(filenames), map_location="cuda"))
                    self.agents[3].actor_network.load_state_dict(torch.load(dir_path+random.choice(filenames), map_location="cuda"))
                    
            self.noise = max(0.05, self.noise - 0.0001)
            self.epsilon = max(0.05, self.epsilon - 0.0001)

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
            year_obs = np.zeros((1,))
            global_obs, private_obs = self.eval_env.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
            cat_gov_state = np.zeros((self.args.prev, self.args.gov_obs_dim+1))
            cat_house_state = np.zeros((self.args.prev, self.envs.households.n_households, (self.args.house_obs_dim - self.args.gov_obs_dim)))
            cat_gov_state[:-1] = cat_gov_state[1:]
            cat_gov_state[-1] = deepcopy(np.concatenate([global_obs, year_obs/self.args.epoch_length], -1))
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
                    gov_reward = 1 / (1 + np.exp(-gov_reward))
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)
                    year_obs[0] += 1.0
                    cat_gov_state[:-1] = cat_gov_state[1:]
                    cat_gov_state[-1] = deepcopy(np.concatenate([next_global_obs, year_obs/self.args.epoch_length], -1))
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
            print(action)
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
