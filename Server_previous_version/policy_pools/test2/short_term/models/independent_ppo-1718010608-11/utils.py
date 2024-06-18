import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions import Distribution


"""
the tanhnormal distributions from rlkit may not stable

"""
class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.cuda = cuda
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        sample_mean = torch.zeros(self.normal_mean.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        sample_std = torch.ones(self.normal_std.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        z = (self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample())
        z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

# get action_infos
class get_action_info:
    def __init__(self, pis, cuda=False):
        self.mean, self.std = pis
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)
    
    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return actions, pretanh 
            else:
                actions = self.dist.sample()
        else:
            actions = torch.tanh(self.mean)
        #return F.softmax(actions, dim=1)
        return actions

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)


# add ounoise here
class ounoise():
    def __init__(self, std, action_dim, mean=0, theta=0.15, dt=1e-2, x0=None):
        self.std = std
        self.mean = mean
        self.action_dim = action_dim
        self.theta = theta
        self.dt = dt
        self.x0 = x0

    # reset the noise
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.action_dim)

    # generate noise
    def noise(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.x_prev = x
        return x


def select_actions(pi):
    mean, std = pi
    actions = Normal(mean, std).sample()
    # return actions
    return actions.detach().cpu().numpy().squeeze()

def evaluate_actions(pi, actions):
    mean, std = pi
    normal_dist = Normal(mean, std)
    log_prob = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
    entropy = normal_dist.entropy().mean()
    return log_prob, entropy


import os

import importlib.util
import sys
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
import pandas as pd
from scipy.special import softmax
def select_policy_from_models_from_server(log_path, env, cfg, temperature=100.0):

    policy_pool = pd.read_csv(log_path)

    # 获取score列
    # scores = policy_pool['score'].values

    # 归一化score列
    # normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # 对归一化的score应用softmax函数
    # probabilities = softmax(normalized_scores / temperature)
    # print("Normalized Scores:", normalized_scores)
    # print("Probabilities:", probabilities)

    # 根据概率分布进行抽样
    np.random.seed()
    selected_index = np.random.choice(policy_pool.index)
    
    selected_index = 0 # for debug
    # print(f"Selected index: {selected_index}")
    selected_policy = policy_pool.loc[selected_index]
    selected_policy_path = selected_policy['path']
    
    agent = load_model(selected_policy_path, env, cfg)
    return agent

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