import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # 共享特征提取层 (MLP)
        # Increased hidden size for larger input (72 dims)
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        
        # Actor Head
        self.actor = layer_init(nn.Linear(128, action_dim), std=0.01)
        
        # Critic Head
        self.critic = layer_init(nn.Linear(128, 1), std=1)
        
    def get_value(self, x):
        return self.critic(self.feature_net(x))
    
    def get_action_and_value(self, x, action_mask=None, action=None):
        hidden = self.feature_net(x)
        logits = self.actor(hidden)
        
        # Apply Action Masking
        # Mask: 1.0 = valid, 0.0 = invalid
        # Invalid logits -> -inf
        if action_mask is not None:
            # logits[mask == 0] = -1e8
            # 更稳健的写法:
            HUGE_NEG = -1e8
            logits = logits + (action_mask - 1.0) * (-HUGE_NEG) 
            # 如果 mask=1, +0; 如果 mask=0, -1e8 (变成极小值)
            
        probs = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
