import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base import BaseAlgorithm
from model import ActorCritic

class PPO(BaseAlgorithm):
    def __init__(self, config, obs_dim, action_dim):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = config.device
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Storage
        self.obs_buffer = []
        self.mask_buffer = []
        self.action_buffer = []
        self.logprob_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.done_buffer = []
        
    def get_action(self, obs, mask=None, deterministic=False):
        """
        Returns:
            action: Tensor
            info: Dict containing logprob, value, entropy
        """
        with torch.no_grad():
            action, logprob, entropy, value = self.model.get_action_and_value(obs, mask)
        
        info = {
            "logprob": logprob,
            "value": value,
            "entropy": entropy
        }
        return action, info

    def get_value(self, obs):
        with torch.no_grad():
            return self.model.get_value(obs)

    def store_transition(self, obs, action, reward, done, mask, info):
        self.obs_buffer.append(obs)
        self.mask_buffer.append(mask)
        self.action_buffer.append(action)
        self.logprob_buffer.append(info["logprob"])
        self.reward_buffer.append(reward)
        self.value_buffer.append(info["value"].flatten())
        self.done_buffer.append(done)

    def update(self, next_obs=None):
        config = self.config
        device = self.device
        
        # Convert buffers to tensors/arrays
        # obs_buffer: List of [B, Dim] -> [T, B, Dim]
        # But we want flattened [T*B, Dim] for training, but [T, B] for GAE
        
        # Check length
        num_steps = len(self.obs_buffer)
        
        # Stack lists
        # Note: All tensors should be on device already if passed correctly from train.py
        b_obs = torch.stack(self.obs_buffer)
        b_mask = torch.stack(self.mask_buffer)
        b_actions = torch.stack(self.action_buffer)
        b_logprobs = torch.stack(self.logprob_buffer)
        b_rewards = torch.stack(self.reward_buffer)
        b_values = torch.stack(self.value_buffer)
        b_dones = torch.stack(self.done_buffer)
        
        # Calculate GAE
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(b_rewards).to(device)
            lastgaelam = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = b_values[t + 1]
                
                # We use b_dones[t] to mask the connection to the future
                nonterminal = 1.0 - b_dones[t].float()
                
                delta = b_rewards[t] + config.gamma * nextvalues * nonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nonterminal * lastgaelam
                
            returns = advantages + b_values

        # Flatten for PPO update
        b_obs = b_obs.reshape((-1, self.obs_dim))
        b_mask = b_mask.reshape((-1, self.action_dim))
        b_logprobs = b_logprobs.reshape(-1)
        b_actions = b_actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = b_values.reshape(-1)

        # Optimizing
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        results = {}
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
                    b_obs[mb_inds], 
                    b_mask[mb_inds], 
                    b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # Normalize Advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
            
            results['loss'] = loss.item()
            results['pg_loss'] = pg_loss.item()
            results['v_loss'] = v_loss.item()
            results['entropy_loss'] = entropy_loss.item()
            results['approx_kl'] = approx_kl.item()
        
        # Clear buffers
        self.obs_buffer = []
        self.mask_buffer = []
        self.action_buffer = []
        self.logprob_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.done_buffer = []
        
        return results

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
