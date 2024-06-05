import torch
from torch.distributions.normal import Normal

def select_actions(pi):

    torch.manual_seed(1) # for debugging

    mean, std = pi
    actions = Normal(mean, std).sample()
    # return actions
    return actions.detach().cpu().numpy().squeeze()


# 批量输入
pi1 = (torch.tensor([[[0.0035, 0.0212],
                      [0.0032, 0.0171],
                      [0.0030, 0.0160],
                      [0.0032, 0.0172]]], device='cuda:0'), 
       torch.tensor([[[1.0004, 0.9999],
                      [1.0004, 0.9999],
                      [1.0004, 0.9999],
                      [1.0004, 0.9999]]], device='cuda:0'))

# 单独输入
pi2 = (torch.tensor([[0.0035, 0.0212]], device='cuda:0'), 
       torch.tensor([[1.0004, 0.9999]], device='cuda:0'))
pi3 = (torch.tensor([[0.0032, 0.0171]], device='cuda:0'), 
       torch.tensor([[1.0004, 0.9999]], device='cuda:0'))
pi4 = (torch.tensor([[0.0030, 0.0160]], device='cuda:0'), 
       torch.tensor([[1.0004, 0.9999]], device='cuda:0'))
pi5 = (torch.tensor([[0.0032, 0.0172]], device='cuda:0'), 
       torch.tensor([[1.0004, 0.9999]], device='cuda:0'))

# 生成动作
actions1 = select_actions(pi1)

# 分开生成动作
actions2 = select_actions(pi2)
actions3 = select_actions(pi3)
actions4 = select_actions(pi4)
actions5 = select_actions(pi5)

print("Batch actions:")
print(actions1)

print("\nSeparate actions:")
print(actions2)
print(actions3)
print(actions4)
print(actions5)
