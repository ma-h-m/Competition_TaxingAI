import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from actor_critic import Actor, Critic
import numpy as np
import copy

class MATD3:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq
        self.actor_pointer = 0
        
        critic_input_size = self.args.gov_action_dim + self.args.house_action_dim * self.args.n_households + (self.args.gov_obs_dim + 1 + (self.args.house_obs_dim-self.args.gov_obs_dim) * self.args.n_households) * args.prev
        # create the network
        if agent_id == self.args.n_agents-1:   # government agent
            self.actor_network = Actor((args.gov_obs_dim+1)*args.prev, args.gov_action_dim, hidden_size=args.hidden_size, step=self.args.prev)
            self.actor_target_network = Actor((args.gov_obs_dim+1)*args.prev, args.gov_action_dim, hidden_size=args.hidden_size, step=self.args.prev)
        else:  # household agent
            self.actor_network = Actor((args.house_obs_dim+1)*args.prev, args.house_action_dim, hidden_size=args.hidden_size, step=self.args.prev)
            self.actor_target_network = Actor((args.house_obs_dim+1)*args.prev, args.house_action_dim, hidden_size=args.hidden_size, step=self.args.prev)
        
        self.critic_network = Critic(critic_input_size+1, hidden_size=args.hidden_size, step=self.args.prev)
        self.critic_target_network = Critic(critic_input_size+1, hidden_size=args.hidden_size, step=self.args.prev)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.p_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.q_lr)
        
        # if use the cuda...
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
    
    def eval_model(self):
        self.actor_network.eval()
        self.critic_network.eval()
        self.actor_target_network.eval()
        self.critic_target_network.eval()
    
    def train_model(self):
        self.actor_network.train()
        self.critic_network.train()
        self.actor_target_network.train()
        self.critic_target_network.train()

    def select_action(self, o, noise_rate, epsilon, curr_id):
        if curr_id == self.args.n_agents-1:
            action_dim = self.args.gov_action_dim
        else:
            action_dim = self.args.house_action_dim
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-1, 1, action_dim)
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.detach().cpu().numpy()
            u = (u + np.random.normal(0, noise_rate, size=action_dim))
            u = np.clip(u, -1, +1)
        return u.copy()

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents, curr_id):
        self.actor_pointer += 1
        global_obs, private_obs, gov_action, hou_action, gov_reward, house_reward, next_global_obs, next_private_obs, done = transitions
        
        global_obses = torch.tensor(global_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        private_obses = torch.tensor(private_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        hou_actions = torch.tensor(hou_action, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_reward, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_rewards = torch.tensor(house_reward, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - done, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)

        if curr_id == self.args.n_agents - 1:  # government agent
            r = gov_rewards.view(-1, 1)
        else:
            r = house_rewards[:, curr_id]
            
        # 用来装每个agent经验中的各项
        o = torch.cat((private_obses.view(self.args.batch_size, -1), global_obses), dim=1)
        u = torch.cat((hou_actions.view(self.args.batch_size, -1), gov_actions), dim=1)
        o_next = torch.cat((next_private_obses.view(self.args.batch_size, -1), next_global_obses), dim=1)

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.args.n_agents - 1:
                    this_next_o = next_global_obses
                else:
                    this_next_o = torch.cat((next_global_obses, next_private_obses[:, agent_id]), dim=1)
                    
                if agent_id == curr_id:
                    batch_a_next = self.actor_target_network(this_next_o)
                    noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    batch_a_next = (batch_a_next + noise).clamp(-1.0, 1.0)
                    u_next.append(batch_a_next)
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    batch_a_next = other_agents[index].actor_target_network(this_next_o)
                    noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    batch_a_next = (batch_a_next + noise).clamp(-1.0, 1.0)
                    u_next.append(batch_a_next)
                    index += 1
            u_next = torch.cat(u_next, dim=1)
            Q1_next, Q2_next = self.critic_target_network(o_next, u_next)
            target_q = (r + self.args.gamma * inverse_dones * torch.min(Q1_next, Q2_next)).detach()

        # the q loss
        current_Q1, current_Q2 = self.critic_network(o, u)
        critic_loss = F.mse_loss(current_Q1, target_q) + F.mse_loss(current_Q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 20.0)
        self.critic_optim.step()
        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        # Trick 3:delayed policy updates
        new_house_actions = copy.copy(hou_actions)
        new_gov_actions = copy.copy(gov_actions)
        if curr_id == self.args.n_agents - 1:
            new_gov_actions = self.actor_network(global_obses)
        else:
            this_o = torch.cat((global_obses, private_obses[:, curr_id]), dim=1)
            new_house_actions[:, curr_id] = self.actor_network(this_o)
            
        u = torch.cat((new_house_actions.view(self.args.batch_size, -1), new_gov_actions), dim=1)
        current_Q1, current_Q2 = self.critic_network(o, u)
        actor_loss = - torch.min(current_Q1, current_Q2).mean()
        if curr_id == self.args.n_agents - 1:
            gov_actions_teacher = np.ones((new_gov_actions.size(0), new_gov_actions.size(1)))
            '''
            gov_actions_teacher[:,0] = -1.0
            gov_actions_teacher[:,1] = 1.0
            gov_actions_teacher[:,2] = 1.0
            gov_actions_teacher[:,3] = 1.0
            gov_actions_teacher[:,4] = -1.0
            '''
            gov_actions_teacher[:,0] = 0.45316147804260254
            gov_actions_teacher[:,1] = 0.07031098008155823
            gov_actions_teacher[:,2] = -0.18397200107574463
            gov_actions_teacher[:,3] = 0.2060547024011612
            gov_actions_teacher[:,4] = 0.29922619462013245
            #'''
            gov_actions_teacher = torch.tensor(gov_actions_teacher, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
            teacher_loss = F.mse_loss(new_gov_actions, gov_actions_teacher)
        else:
            hou_actions_teacher = np.ones((new_house_actions.size(0), 2))*(-0.6179999709129333)
            hou_actions_teacher = torch.tensor(hou_actions_teacher, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
            teacher_loss = F.mse_loss(new_house_actions[:, curr_id], hou_actions_teacher)
        actor_loss = actor_loss + 0.25 * teacher_loss
        if self.actor_pointer % self.policy_update_freq == 0:
            # update the network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 20.0)
            self.actor_optim.step()
            self._soft_update_target_network()
        self.train_step += 1
        
        return actor_loss.item(), critic_loss.item()

