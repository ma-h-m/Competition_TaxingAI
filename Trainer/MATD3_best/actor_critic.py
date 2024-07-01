import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, step=5):
        # input_size (B,70) (B,90)
        super(Actor, self).__init__()
        self.step = step
        self.one_step = input_size//step
        self.split = 4
        self.fc1 = nn.Linear(self.one_step, self.split)
        self.fc2 = nn.Linear(self.split * self.step + self.one_step, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.action_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view((x.size(0), self.step, -1))
        steps = []
        for i in range(self.step):
            steps.append(F.dropout(F.leaky_relu(self.fc1(x[:,i])), p=0.5))
        steps.append(x[:,-1])
        x = torch.cat(steps, -1)
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = F.leaky_relu(self.fc3(x))
        x = F.dropout(x, p=0.5)
        actions = torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=128, step=5):
        super(Critic, self).__init__()
        self.step = step
        self.one_step = (input_size-13)//step
        self.split = 4
        self.fc1 = nn.Linear(self.one_step, self.split)
        self.fc2 = nn.Linear(self.split * self.step + self.one_step + 13, hidden_size)
        self.fc3_1 = nn.Linear(hidden_size, hidden_size)
        self.q_out_1 = nn.Linear(hidden_size, 1)
        self.fc3_2 = nn.Linear(hidden_size, hidden_size)
        self.q_out_2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = state.view((state.size(0), self.step, -1))
        steps = []
        for i in range(self.step):
            steps.append(F.dropout(F.relu(self.fc1(x[:,i])), p=0.5))
        steps.append(x[:,-1])
        steps.append(action)
        x = torch.cat(steps, dim=1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x1 = F.relu(self.fc3_1(x))
        x1 = F.dropout(x1, p=0.5)
        q_value_1 = self.q_out_1(x1)
        x2 = F.relu(self.fc3_2(x))
        x2 = F.dropout(x2, p=0.5)
        q_value_2 = self.q_out_2(x2)
        return q_value_1, q_value_2
    
    def Q1(self, state, action):
        x = state.view((state.size(0), self.step, -1))
        steps = []
        for i in range(self.step):
            steps.append(F.dropout(F.relu(self.fc1(x[:,i])), p=0.5))
        steps.append(x[:,-1])
        steps.append(action)
        x = torch.cat(steps, dim=1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x1 = F.relu(self.fc3_1(x))
        x1 = F.dropout(x1, p=0.5)
        q_value_1 = self.q_out_1(x1)
        return q_value_1


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.1)
        nn.init.constant_(m.bias, 0)