import torch
import torch.nn as nn
import torch.nn.functional as F
from tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

class Actor(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.5):
        # input_size (B,70) (B,90)
        super(Actor, self).__init__()
        self.step = input_size
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.action_out = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        inputs = x.view((x.size(0), self.step, -1))
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        actions = torch.tanh(self.action_out(y1[:, :, -1]))
        return actions

class Critic(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.5):
        super(Critic, self).__init__()
        self.step = input_size
        hidden_size = 256
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]+13, hidden_size)
        self.q_out_1 = nn.Linear(hidden_size, output_size)
        self.q_out_2 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        inputs = state.view((state.size(0), self.step, -1))
        x = self.tcn(inputs)  # input should have dimension (N, C, L)
        x = torch.cat([x[:, :, -1], action], -1)
        x = F.relu(self.linear(x))
        q_value_1 = self.q_out_1(x)
        q_value_2 = self.q_out_2(x)
        return q_value_1, q_value_2
    
    def Q1(self, state, action):
        inputs = state.view((state.size(0), self.step, -1))
        x = self.tcn(inputs)  # input should have dimension (N, C, L)
        x = torch.cat([x[:, :, -1], action], -1)
        x = F.relu(self.linear(x))
        q_value_1 = self.q_out_1(x)
        return q_value_1


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.1)
        nn.init.constant_(m.bias, 0)