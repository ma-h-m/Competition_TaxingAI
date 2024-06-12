# independent ppo

import numpy as np
import torch
from torch import optim
import os,sys
sys.path.append(os.path.abspath('../..'))
from .models import mlp_net                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
from .utils import select_actions, evaluate_actions, select_policy_from_models_from_server
from .log_path import make_logpath
from datetime import datetime
from env.evaluation import save_parameters
import os
import copy
import wandb
import pickle
import time
from .model_pools_update import update_short_term_policy_pool, update_long_term_policy_pool#, update_top_k_policy_pool
from .client import initial_communicate_with_server, push_folder, fetch_random_models

def load_params_from_file(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

def fetch_data(alg, i):
    path = "/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/100/"+ alg+"/epoch_0_step_%d_100_gdp_parameters.pkl"%(i+1)
    para = load_params_from_file(path)
    return para['valid_action_dict']['Household']

from omegaconf import OmegaConf
def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    
    OmegaConf.save(args, os.path.dirname(path) + '/hyper_params.yaml')
    


class agent:
    def __init__(self,  envs, args, algo_name = "independent_ppo", house_net_path = None, gov_net_path = None, test = False):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        # start to build the network.
        self.households_net = mlp_net(self.envs.households.observation_space.shape[0], self.envs.households.action_space.shape[1])
        self.households_old_net = copy.deepcopy(self.households_net)
        
        self.gov_net = mlp_net(self.envs.government.observation_space.shape[0], self.envs.government.action_space.shape[0])
        self.gov_old_net = copy.deepcopy(self.gov_net)
        # if use the cuda...
        if self.args.cuda:
            self.households_net.cuda()
            self.households_old_net.cuda()
            self.gov_net.cuda()
            self.gov_old_net.cuda()
        # define the optimizer...
        if not test:
            self.house_optimizer = optim.Adam(self.households_net.parameters(), self.args.p_lr, eps=self.args.eps)
            self.gov_optimizer = optim.Adam(self.gov_net.parameters(), self.args.p_lr, eps=self.args.eps)
            
        # Load pre-trained weights if provided
        if house_net_path:
            self.load_pretrained_weights(house_net_path, "households")
            self.house_net_path = house_net_path
        if gov_net_path:
            self.load_pretrained_weights(gov_net_path, "government")
            self.gov_net_path = gov_net_path


        # get the observation
        # self.batch_ob_shape = (self.args.n_households * self.args.epoch_length, ) + self.envs.households.observation_space.shape
        self.dones = np.tile(False, (self.args.n_households, 1))
        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]

        if not test:
            self.model_path, _ = make_logpath(algo=algo_name,n=self.args.n_households)

        self.identifier = algo_name + "-" + str(int(time.time()))
        self.algo_name = algo_name
        if not test:
            save_args(path=self.model_path, args=self.args)
        
        # wandb.init(
        #     # config=self.args,
        #     project="AI_TaxingPolicy",
        #     # entity="ai_tax",
        #     # name=self.model_path.parent.parent.name + "-" + self.model_path.name + '  n=' + str(self.args.n_households),
        #     # dir=str(self.model_path),
        #     job_type="training",
        #     reinit=True
        # )
        
    def load_pretrained_weights(self, model_path, agent_type):
        if agent_type == "households":
            self.households_net.load_state_dict(torch.load(model_path))
            self.households_old_net = copy.deepcopy(self.households_net)
        elif agent_type == "government":
            self.gov_net.load_state_dict(torch.load(model_path))
            self.gov_old_net = copy.deepcopy(self.gov_net)
        else:
            raise ValueError("Unknown agent type. Choose 'households' or 'government'.")

    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    def obs_concate(self, global_obs, private_obs, update=True):
        if update == True:
            global_obs = global_obs.unsqueeze(1)
        n_global_obs = global_obs.repeat(1, self.args.n_households, 1)
        return torch.cat([n_global_obs, private_obs], dim=-1).flatten(0,1)

    def obs_concate_numpy(self, global_obs, private_obs, update=True):
        # if update == True:
        #     global_obs = global_obs.unsqueeze(1)
        n_global_obs = np.tile(global_obs, (self.args.n_households, 1))
        return np.concatenate((n_global_obs, private_obs), axis=-1)

    # def action_wrapper(self, actions):
    #     return (actions - np.min(actions, axis=0)) / (np.max(actions, axis=0) - np.min(actions, axis=0))
    def action_wrapper(self, actions):
        return np.clip(actions, 0, 1)
    def mb_data_process(self, mb_obs, mb_rewards, mb_actions, mb_dones, mb_values, agent, isOneHousehold=False):
    
        # process the rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_values = np.asarray(mb_values, dtype=np.float32)
    
        with torch.no_grad():
            if agent == "households":
                if isOneHousehold:
                    n_agent = 1
                    obs_tensor = self._get_tensor_inputs(self.obs[self.args.n_households - 1])
                    last_values, _ = self.households_net(obs_tensor)
                else:
                    n_agent = self.args.n_households
                    obs_tensor = self._get_tensor_inputs(self.obs)
                    last_values, _ = self.households_net(obs_tensor)
            else:
                n_agent = 1
                obs_tensor = self._get_tensor_inputs(self.obs[0][:self.envs.government.observation_space.shape[0]])
                last_values, _ = self.gov_net(obs_tensor)
            last_values = last_values.detach().cpu().numpy().squeeze(0)
        # start to compute advantages...
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.args.epoch_length)):
            if t == self.args.epoch_length - 1:
                nextnonterminal = 1.0 - np.asarray(self.dones[:n_agent])
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
        
            delta = mb_rewards[t] + self.args.ppo_gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.args.ppo_gamma * self.args.ppo_tau * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return mb_obs, mb_actions, mb_returns, mb_advs
    
    def observation_wrapper(self, global_obs, private_obs):
        # global
        global_obs[0] /= 1e7
        global_obs[1] /= 1e5
        global_obs[3] /= 1e5
        global_obs[4] /= 1e5
        private_obs[:, 1] /= 1e5
        return global_obs, private_obs
    def observation_wrapper_with_only_one_houselhold(self, global_obs, private_obs):
        # global
        # global_obs[0] /= 1e7
        # global_obs[1] /= 1e5
        # global_obs[3] /= 1e5
        # global_obs[4] /= 1e5
        # private_obs[1] /= 1e5
        tmp_global_obs = global_obs.copy()
        tmp_private_obs = private_obs.copy()
        tmp_global_obs[0] /= 1e7
        tmp_global_obs[1] /= 1e5
        tmp_global_obs[3] /= 1e5
        tmp_global_obs[4] /= 1e5
        tmp_private_obs[1] /= 1e5
        return tmp_global_obs, tmp_private_obs
        
# for evaluation with mutiple kinds of agents
    def get_one_action(self, global_obs, private_obs, isHousehold=True):
        global_obs, private_obs = self.observation_wrapper_with_only_one_houselhold(global_obs, private_obs)
        with torch.no_grad():
            if isHousehold:
                obs = np.concatenate((global_obs, private_obs), axis=-1)
                obs = self._get_tensor_inputs(obs)
                values, pis = self.households_net(obs)
                action = select_actions(pis)
                action = self.action_wrapper(action)
                action = self.hou_action_max * (action * 2 - 1)
            else:
                values, pis = self.gov_net(self._get_tensor_inputs(global_obs))
                action = select_actions(pis)
                action = self.action_wrapper(action)
                action = self.gov_action_max * (action * 2 - 1)
        
        return action
    
    def select_actions(self, pis, isHousehold=True):
        action = select_actions(pis)
        action = self.action_wrapper(action)
        if isHousehold:
            action = self.hou_action_max * (action * 2 - 1)
        else:
            action = self.gov_action_max * (action * 2 - 1)
        return action
    
    
    def get_one_pis(self, global_obs, private_obs, isHousehold=True):
        global_obs, private_obs = self.observation_wrapper_with_only_one_houselhold(global_obs, private_obs)
        with torch.no_grad():
            if isHousehold:
                obs = np.concatenate((global_obs, private_obs), axis=-1)
                obs = self._get_tensor_inputs(obs)
                values, pis = self.households_net(obs)

            else:
                values, pis = self.gov_net(self._get_tensor_inputs(global_obs))

        
        return pis
    

# could be eigher models from server or models from short/long term policy pool
    def train_with_external_models(self, external_household_logs, external_government_logs):
        episode_rewards = np.zeros((self.args.n_households, ), dtype=np.float32)
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)

        hh_num = self.args.n_households
        external_government_model = select_policy_from_models_from_server(external_government_logs, self.envs, self.args)
        external_household_agents = [select_policy_from_models_from_server(external_household_logs, self.envs, self.args) for _ in range(hh_num)]

        # 先写自己的government + 外来的household
        print("Training with external household models...")
        for update in range(10):




            gov_mb_obs, gov_mb_rewards, gov_mb_actions, gov_mb_dones, gov_mb_values = [], [], [], [], []
            
            self._adjust_learning_rate(update, 10)

            for step in range(self.args.epoch_length):

                
                with torch.no_grad():
                    # get tensors
                    gov_values, gov_pis = self.gov_net(self._get_tensor_inputs(global_obs))
                    # house_values, house_pis = self.households_net(self._get_tensor_inputs(self.obs))
                    household_actions = []
                    for hh_agent, private_obs in zip(external_household_agents, private_obs):
                        tmp_action = hh_agent.get_one_action(global_obs, private_obs, isHousehold=True)
                        household_actions.append(tmp_action)

                    
                # select actions
                gov_actions = select_actions(gov_pis)
                gov_action = self.action_wrapper(gov_actions)
                gov_action = self.gov_action_max * (gov_action * 2 - 1)
                # house_actions = select_actions(house_pis)
                # input_actions = self.action_wrapper(house_actions)

                # household_actions = household_agents[0].select_actions((mean_tensor, std_tensor))
                





                action = {self.envs.government.name: np.array(gov_action),
                          self.envs.households.name: np.array(household_actions) }

                # gov data add
                gov_mb_obs.append(np.copy(global_obs))
                gov_mb_actions.append(gov_actions)
                gov_mb_values.append(gov_values.detach().cpu().numpy().squeeze(0))
                
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                self.dones = np.tile(done, (self.args.n_households, 1))

                gov_mb_dones.append(done)
                gov_mb_rewards.append(gov_reward.reshape(1))
                # clear the observation
                if done:
                    next_global_obs, next_private_obs = self.envs.reset()
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                self.obs = self.obs_concate_numpy(next_global_obs, next_private_obs, update=False)
                global_obs, private_obs = next_global_obs, next_private_obs

                # process the rewards part -- display the rewards on the screen
                episode_rewards += house_reward.flatten()
                masks = np.array([0.0 if done_ else 1.0 for done_ in self.dones], dtype=np.float32)

                episode_rewards *= masks
            
            # after compute the returns, let's process the rollouts
            
            gov_mb_obs, gov_mb_actions, gov_mb_returns, gov_mb_advs = self.mb_data_process(gov_mb_obs, gov_mb_rewards, gov_mb_actions, gov_mb_dones, gov_mb_values, agent="government")
            
            self.gov_old_net.load_state_dict(self.gov_net.state_dict())
            # start to update the network
            self._update_gov_network(gov_mb_obs, gov_mb_actions, gov_mb_returns, gov_mb_advs)

        # 再写的是自己的household + 外来的household + 外来的government(假定household内是对称的，默认把自己的放到第一个，替换掉抽取的最后一个)
        print("Training with external government models...")
        for update in range(10):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            self._adjust_learning_rate(update, 10)

            for step in range(self.args.epoch_length):
                with torch.no_grad():
                    # get tensors
                    gov_action = external_government_model.get_one_action(global_obs, private_obs, isHousehold=False)
                    
                    household_actions = []

                    obs = np.concatenate((global_obs, private_obs[hh_num - 1]), axis=-1)
                    obs_in_tensor = self._get_tensor_inputs(obs)
                    house_values, house_pis = self.households_net(obs_in_tensor)
                    self_action = select_actions(house_pis)
                    self_action = self.action_wrapper(self_action)
                    self_action = self.hou_action_max * (self_action * 2 - 1)

                    for index in range(hh_num - 1):
                        tmp_action = external_household_agents[index].get_one_action(global_obs, private_obs[index], isHousehold=True)
                        household_actions.append(tmp_action)
                    
                    household_actions.append(self_action)

                    
                # select actions

                # house_actions = select_actions(house_pis)
                # input_actions = self.action_wrapper(house_actions)

                # household_actions = household_agents[0].select_actions((mean_tensor, std_tensor))
                





                action = {self.envs.government.name: np.array(gov_action),
                          self.envs.households.name: np.array(household_actions) }

                # start to store information
                mb_obs.append(np.copy(obs))
                mb_actions.append(self_action)
                mb_values.append(house_values.detach().cpu().numpy().squeeze(0))
                
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                self.dones = np.tile(done, (self.args.n_households, 1))

                mb_dones.append(done)
                mb_rewards.append(house_reward[hh_num - 1])
                # clear the observation
                if done:
                    next_global_obs, next_private_obs = self.envs.reset()
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                self.obs = self.obs_concate_numpy(next_global_obs, next_private_obs, update=False)
                global_obs, private_obs = next_global_obs, next_private_obs

                # process the rewards part -- display the rewards on the screen
                episode_rewards += house_reward[hh_num - 1].flatten()
                masks = np.array([0.0 if done_ else 1.0 for done_ in self.dones], dtype=np.float32)

                episode_rewards *= masks
            
            # after compute the returns, let's process the rollouts
            
            mb_obs, mb_actions, mb_returns, mb_advs = self.mb_data_process(mb_obs, mb_rewards, mb_actions, mb_dones, mb_values, agent="households", isOneHousehold=True)
          
            self.households_old_net.load_state_dict(self.households_net.state_dict())
      
            # start to update the network
            self._update_house_network(mb_obs, mb_actions, mb_returns, mb_advs)




    # start to train the network...
    def learn(self):
        episode_rewards = np.zeros((self.args.n_households, ), dtype=np.float32)
        global_obs, private_obs = self.envs.reset()
        global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
        gov_rew = []
        house_rew = []
        epochs = []

        for update in range(self.args.n_epochs):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            gov_mb_obs, gov_mb_rewards, gov_mb_actions, gov_mb_dones, gov_mb_values = [], [], [], [], []
            self._adjust_learning_rate(update, self.args.n_epochs)
            for step in range(self.args.epoch_length):
                with torch.no_grad():
                    # get tensors
                    gov_values, gov_pis = self.gov_net(self._get_tensor_inputs(global_obs))
                    house_values, house_pis = self.households_net(self._get_tensor_inputs(self.obs))
                    
                # select actions
                gov_actions = select_actions(gov_pis)
                gov_action = self.action_wrapper(gov_actions)
                house_actions = select_actions(house_pis)
                input_actions = self.action_wrapper(house_actions)

                action = {self.envs.government.name: self.gov_action_max * (gov_action * 2 - 1),
                          self.envs.households.name: self.hou_action_max * (input_actions*2-1) }
                
                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(house_actions)
                mb_values.append(house_values.detach().cpu().numpy().squeeze(0))
                # gov data add
                gov_mb_obs.append(np.copy(global_obs))
                gov_mb_actions.append(gov_actions)
                gov_mb_values.append(gov_values.detach().cpu().numpy().squeeze(0))
                
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                self.dones = np.tile(done, (self.args.n_households, 1))
                mb_dones.append(self.dones)
                mb_rewards.append(house_reward)
                gov_mb_dones.append(done)
                gov_mb_rewards.append(gov_reward.reshape(1))
                # clear the observation
                if done:
                    next_global_obs, next_private_obs = self.envs.reset()
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                self.obs = self.obs_concate_numpy(next_global_obs, next_private_obs, update=False)
                # process the rewards part -- display the rewards on the screen
                episode_rewards += house_reward.flatten()
                masks = np.array([0.0 if done_ else 1.0 for done_ in self.dones], dtype=np.float32)

                episode_rewards *= masks
            
            # after compute the returns, let's process the rollouts
            mb_obs, mb_actions, mb_returns, mb_advs = self.mb_data_process(mb_obs, mb_rewards, mb_actions, mb_dones, mb_values, agent="households")
            gov_mb_obs, gov_mb_actions, gov_mb_returns, gov_mb_advs = self.mb_data_process(gov_mb_obs, gov_mb_rewards, gov_mb_actions, gov_mb_dones, gov_mb_values, agent="government")
            
            self.households_old_net.load_state_dict(self.households_net.state_dict())
            self.gov_old_net.load_state_dict(self.gov_net.state_dict())
            # start to update the network
            house_policy_loss, house_value_loss, house_ent_loss, gov_policy_loss, gov_value_loss, gov_ent_loss =\
                self._update_network(mb_obs, mb_actions, mb_returns, mb_advs, gov_mb_obs, gov_mb_actions, gov_mb_returns, gov_mb_advs)
            
        # training with external models

            self.train_with_external_models('TaxAI_modified/agents/model_pools/models_from_server/log_household.csv', 'TaxAI_modified/agents/model_pools/models_from_server/log_government.csv')


            if update % self.args.display_interval == 0:
                # start to do the evaluation
                mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
                # store rewards and step
                now_step = (update + 1) * self.args.epoch_length
                gov_rew.append(mean_gov_rewards)
                house_rew.append(mean_house_rewards)
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                # np.savetxt(str(self.model_path) + "/wealth_stack.txt", wealth_stack)
                # np.savetxt(str(self.model_path) + "/income_stack.txt", income_stack)

                # GDP + mean utility + wealth distribution + income distribution
                # wandb.log({"mean households utility": mean_house_rewards,
                #            "goverment utility": mean_gov_rewards,
                #            "wealth gini": avg_wealth_gini,
                #            "income gini": avg_income_gini,
                #            "GDP": avg_gdp,
                #            "years": years,
                #            "tax per households": avg_mean_tax,
                #            "post income per households": avg_mean_post_income,
                #            "wealth per households": avg_mean_wealth,
                #            "households actor loss": house_policy_loss,
                #            "households critic loss": house_value_loss,
                #            "gov actor loss": gov_policy_loss,
                #            "gov critic loss": gov_value_loss,
                #            "steps": now_step})
                print('[{}] Update: {} / {}, Frames: {}, Gov_Rewards: {:.3f}, House_Rewards: {:.3f}, years: {:.3f}, PL: {:.3f},'\
                    'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, self.args.n_epochs, now_step, mean_gov_rewards, mean_house_rewards, years, house_policy_loss, house_value_loss, house_ent_loss))

                torch.save(self.households_net.state_dict(), str(self.model_path) + '/house_net.pt')
                torch.save(self.gov_net.state_dict(), str(self.model_path) + '/gov_net.pt')
                # save the model
                update_short_term_policy_pool(self.args.short_term_policy_pool_size, epoch=update, government_score=mean_gov_rewards, household_score=mean_house_rewards, algo= self.algo_name, id=self.identifier)
                
                if update % self.args.long_term_policy_update_freq == 0:
                    update_long_term_policy_pool(self.args.long_term_policy_pool_size, epoch=update, government_score=mean_gov_rewards, household_score=mean_house_rewards, algo= self.algo_name, id=self.identifier)
                
                # if update % self.args.freq_of_pushing_moodels_to_server == 0:
                #     push_folder(self.model_path, self.identifier)
                    
                # if update % self.args.freq_of_fetching_random_models == 0:
                #     fetch_random_models(gov_model_num = 1, household_model_num = 4, dest_dir="TaxAI_modified/agents/model_pools/models_from_server")


                
        # wandb.finish()

    def test(self):
        # self.households_net.load_state_dict(torch.load("/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/10/run2/house_net.pt"))
        # self.gov_net.load_state_dict(torch.load("/home/mqr/code/AI-TaxingPolicy/agents/models/independent_ppo/100/run2/gov_net.pt"))
        mean_gov_rewards, mean_house_rewards, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, avg_wealth_gini, years = self._evaluate_agent()
        print("mean gov reward:", mean_gov_rewards)


    # update the network
    def sub_update_network(self, obs, actions, returns, advantages, inds, nbatch_train, start, agent):
        # get the mini-batchs
        end = start + nbatch_train
        mbinds = inds[start:end]
        mb_obs = obs[mbinds]
        mb_actions = actions[mbinds]
        mb_returns = returns[mbinds]
        mb_advs = advantages[mbinds]
        # convert minibatches to tensor
        mb_obs = self._get_tensor_inputs(mb_obs)
        mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
        mb_returns = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(0)
        mb_advs = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(0)
        # normalize adv
        mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
        if self.args.cuda:
            mb_actions = mb_actions.cuda()
            mb_returns = mb_returns.cuda()
            mb_advs = mb_advs.cuda()
        # start to get values
        if agent == "households":
            mb_values, pis = self.households_net(mb_obs)
            optimizer = self.house_optimizer
            net = self.households_net
        else:
            mb_values, pis = self.gov_net(mb_obs)
            optimizer = self.gov_optimizer
            net = self.gov_net
        # start to calculate the value loss...
        value_loss = (mb_returns - mb_values).pow(2).mean()
        # start to calculate the policy loss
        with torch.no_grad():
            if agent == "households":
                _, old_pis = self.households_old_net(mb_obs)
            else:
                _, old_pis = self.gov_old_net(mb_obs)
            # get the old log probs
            old_log_prob, _ = evaluate_actions(old_pis, mb_actions)
            old_log_prob = old_log_prob.detach()
        # evaluate the current policy
        log_prob, ent_loss = evaluate_actions(pis, mb_actions)
        prob_ratio = torch.exp(log_prob - old_log_prob)
        # surr1
        surr1 = prob_ratio * mb_advs
        surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
        policy_loss = -torch.min(surr1, surr2).mean()
        # final total loss
        total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
        # clear the grad buffer
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.max_grad_norm)
        # update
        optimizer.step()
        
        return policy_loss, value_loss, ent_loss
    def _update_network(self, house_obs, house_actions, house_returns, house_advantages, gov_obs, gov_actions, gov_returns, gov_advantages):
        inds = np.arange(house_obs.shape[0])
        nbatch_train = self.args.batch_size
        for _ in range(self.args.update_epoch):
            np.random.shuffle(inds)
            for start in range(0, house_obs.shape[0], nbatch_train):
                house_policy_loss, house_value_loss, house_ent_loss = self.sub_update_network(house_obs, house_actions, house_returns, house_advantages,  inds, nbatch_train, start, agent="households")
                gov_policy_loss, gov_value_loss, gov_ent_loss = self.sub_update_network(gov_obs, gov_actions, gov_returns, gov_advantages,  inds, nbatch_train, start, agent="government")
        return house_policy_loss.item(), house_value_loss.item(), house_ent_loss.item(), gov_policy_loss.item(), gov_value_loss.item(), gov_ent_loss.item()
    

    def _update_gov_network(self, gov_obs, gov_actions, gov_returns, gov_advantages): # TODO: check whether gov_obs.shape[0] == gov_actions.shape[0]
        inds = np.arange(gov_obs.shape[0])
        nbatch_train = self.args.batch_size
        for _ in range(self.args.update_epoch):
            np.random.shuffle(inds)
            for start in range(0, gov_obs.shape[0], nbatch_train):
                gov_policy_loss, gov_value_loss, gov_ent_loss = self.sub_update_network(gov_obs, gov_actions, gov_returns, gov_advantages,  inds, nbatch_train, start, agent="government")
        return gov_policy_loss.item(), gov_value_loss.item(), gov_ent_loss.item()
    
    def _update_house_network(self, house_obs, house_actions, house_returns, house_advantages):
        inds = np.arange(house_obs.shape[0])
        nbatch_train = self.args.batch_size
        for _ in range(self.args.update_epoch):
            np.random.shuffle(inds)
            for start in range(0, house_obs.shape[0], nbatch_train):
                house_policy_loss, house_value_loss, house_ent_loss = self.sub_update_network(house_obs, house_actions, house_returns, house_advantages,  inds, nbatch_train, start, agent="households")
        return house_policy_loss.item(), house_value_loss.item(), house_ent_loss.item()
    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.p_lr * lr_frac
        for param_group in self.house_optimizer.param_groups:
             param_group['lr'] = adjust_lr
    #
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
                    action = self._evaluate_get_action(global_obs, private_obs)
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs, next_private_obs = self.observation_wrapper(next_global_obs, next_private_obs)

                step_count += 1
                episode_gov_reward += gov_reward
                episode_mean_house_reward += np.mean(house_reward)
                episode_mean_tax.append(np.mean(self.eval_env.tax_array))
                episode_mean_wealth.append(np.mean(self.eval_env.households.at_next))
                episode_mean_post_income.append(np.mean(self.eval_env.post_income))
                episode_gdp.append(self.eval_env.per_household_gdp)
                episode_income_gini.append(self.eval_env.income_gini)
                episode_wealth_gini.append(self.eval_env.wealth_gini)
                # if step_count == 1 or step_count == 100 or step_count == 200 or step_count == 300:
                    # save_parameters(self.model_path, step_count, epoch_i, self.eval_env)
                if done:
                    break

                global_obs = next_global_obs
                private_obs = next_private_obs

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
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
        gov_values, gov_pis = self.gov_net(self._get_tensor_inputs(global_obs))
        house_values, house_pis = self.households_net(self._get_tensor_inputs(self.obs))
        # select actions
        gov_actions = select_actions(gov_pis)
        gov_action = self.action_wrapper(gov_actions)
        house_actions = select_actions(house_pis)
        input_actions = self.action_wrapper(house_actions)

        action = {self.envs.government.name: self.gov_action_max * (gov_action * 2 - 1),
                  self.envs.households.name: self.hou_action_max * (input_actions * 2 - 1)}
        return action
    
    def _evaluate_get_pis(self, global_obs, private_obs):
        self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
        gov_values, gov_pis = self.gov_net(self._get_tensor_inputs(global_obs))
        house_values, house_pis = self.households_net(self._get_tensor_inputs(self.obs))
        
        return gov_pis, house_pis
    # def _evaluate_agent(self):
    #     np.random.seed(1)
    #     total_gov_reward = 0
    #     total_house_reward = 0
    #     total_steps = 0
    #     mean_tax = 0
    #     mean_wealth = 0
    #     mean_post_income = 0
    #     gdp = 0
    #     income_gini = 0
    #     wealth_gini = 0
    #     # for epoch_i in range(self.args.eval_episodes):
    #     for epoch_i in range(1):
    #         global_obs, private_obs = self.eval_env.reset()
    #         episode_gov_reward = 0
    #         episode_mean_house_reward = 0
    #         step_count = 0
    #         episode_mean_tax = []
    #         episode_mean_wealth = []
    #         episode_mean_post_income = []
    #         episode_gdp = []
    #         episode_income_gini = []
    #         episode_wealth_gini = []
    #
    #         while True:
    #             if step_count > 4:
    #                 break
    #             with torch.no_grad():
    #                 # action = self._evaluate_get_action(global_obs, private_obs)
    #                 action = self.test_evaluate_get_action(global_obs, private_obs, step_count)
    #                 next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
    #
    #             step_count += 1
    #             episode_gov_reward += gov_reward
    #             episode_mean_house_reward += np.mean(house_reward)
    #             episode_mean_tax.append(np.mean(self.eval_env.tax_array))
    #             episode_mean_wealth.append(np.mean(self.eval_env.households.at_next))
    #             episode_mean_post_income.append(np.mean(self.eval_env.post_income))
    #             episode_gdp.append(self.eval_env.per_household_gdp)
    #             episode_income_gini.append(self.eval_env.income_gini)
    #             episode_wealth_gini.append(self.eval_env.wealth_gini)
    #             if done:
    #                 break
    #             # if step_count == 1 or step_count == 10 or step_count == 30 or step_count == 300:
    #             save_parameters(self.model_path, step_count, epoch_i, self.eval_env)
    #
    #
    #             global_obs = next_global_obs
    #             private_obs = next_private_obs
    #
    #         total_gov_reward += episode_gov_reward
    #         total_house_reward += episode_mean_house_reward
    #         total_steps += step_count
    #         mean_tax += np.mean(episode_mean_tax)
    #         mean_wealth += np.mean(episode_mean_wealth)
    #         mean_post_income += np.mean(episode_mean_post_income)
    #         gdp += np.mean(episode_gdp)
    #         income_gini += np.mean(episode_income_gini)
    #         wealth_gini += np.mean(episode_wealth_gini)
    #
    #     avg_gov_reward = total_gov_reward / self.args.eval_episodes
    #     avg_house_reward = total_house_reward / self.args.eval_episodes
    #     mean_step = total_steps / self.args.eval_episodes
    #     avg_mean_tax = mean_tax / self.args.eval_episodes
    #     avg_mean_wealth = mean_wealth / self.args.eval_episodes
    #     avg_mean_post_income = mean_post_income / self.args.eval_episodes
    #     avg_gdp = gdp / self.args.eval_episodes
    #     avg_income_gini = income_gini / self.args.eval_episodes
    #     avg_wealth_gini = wealth_gini / self.args.eval_episodes
    #     return avg_gov_reward, avg_house_reward, avg_mean_tax, avg_mean_wealth, avg_mean_post_income, avg_gdp, avg_income_gini, \
    #            avg_wealth_gini, mean_step
    #
    #
    #
    # def test_evaluate_get_action(self, global_obs, private_obs, i):
    #     self.obs = self.obs_concate_numpy(global_obs, private_obs, update=False)
    #
    #     house_values, house_pis = self.households_net(self._get_tensor_inputs(self.obs))
    #     # select actions
    #     # gov_action = np.random.random(self.envs.government.action_space.shape[0])
    #     gov_actions = np.array([[0.99860359, 0.28571885, 0.49025352, 0.59911031, 0.189],
    #                     [0.89874693, 0.716929,   0.49025352, 0.59911031, 0.189],
    #                     [0.032421, 0.3282561 , 0.49025352, 0.59911031, 0.189],
    #                     [0.010699,   0.55726823, 0.49025352 ,0.59911031, 0.189],
    #                     [0.76172986, 0.24048432, 0.49025352, 0.59911031, 0.189]])
    #     gov_action = gov_actions[i]
    #     print(gov_action)
    #
    #     # ppo
    #     house_actions = select_actions(house_pis)
    #     input_actions = self.action_wrapper(house_actions)
    #     print(input_actions)
    #     # random
    #     # temp = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
    #     # input_actions = temp
    #     # temp = np.zeros((self.args.n_households, self.envs.households.action_space.shape[1]))
    #     # temp[:, 0] = 0.75
    #     # temp[:, 1] = 2 / 3
    #     # ga
    #
    #     # input_actions = fetch_data("ga", i)
    #
    #     action = {self.envs.government.name: self.gov_action_max * (gov_action * 2 - 1),
    #               self.envs.households.name: self.hou_action_max * input_actions}
    #     return action
    #
