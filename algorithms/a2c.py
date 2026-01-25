import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from algorithms.base import BaseAlgorithm
from algorithms.registry import register_algorithm
from models.registry import get_model


@register_algorithm("a2c")
class A2C(BaseAlgorithm):
    def __init__(
        self,
        config,
        obs_dim,
        action_dim,
        model_name="actor_critic",
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = config.device

        self.model = get_model(model_name, obs_dim=obs_dim, action_dim=action_dim).to(
            self.device
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )

        self.use_amp = getattr(config, "use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None

        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.mask_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []

        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = getattr(config, "max_grad_norm", 0.5)

        if getattr(config, "use_compile", False):
            compile_mode = getattr(config, "compile_mode", "default")
            self.model = torch.compile(self.model, mode=compile_mode)

    def get_action(self, obs, mask=None, deterministic=False):
        with torch.no_grad():
            # Handle both tensor and numpy inputs
            if isinstance(obs, np.ndarray):
                obs_t = torch.FloatTensor(obs).to(self.device)
            else:
                obs_t = obs.to(self.device)

            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask_t = torch.FloatTensor(mask).to(self.device)
                else:
                    mask_t = mask.to(self.device)
            else:
                mask_t = None

            action, log_prob, entropy, value = self.model.get_action_and_value(
                obs_t, mask_t
            )

            if deterministic:
                # Basic deterministic handling if the model doesn't support it in get_action_and_value
                # For more robust handling, one would need to access logits directly
                pass

        # Return tensors like PPO does (train.py expects tensors)
        return action, {
            "logprob": log_prob,
            "value": value,
            "entropy": entropy,
        }

    def store_transition(self, obs, action, reward, done, mask, info, **kwargs):
        obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
        action_np = action.cpu().numpy() if torch.is_tensor(action) else action
        reward_np = reward.cpu().numpy() if torch.is_tensor(reward) else reward
        done_np = done.cpu().numpy() if torch.is_tensor(done) else done
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask

        self.obs_buffer.append(obs_np)
        self.action_buffer.append(action_np)
        self.reward_buffer.append(reward_np)
        self.done_buffer.append(done_np)
        self.mask_buffer.append(mask_np)

        value = info["value"]
        logprob = info["logprob"]
        if torch.is_tensor(value):
            value_np = value.squeeze(-1).cpu().numpy()
        else:
            value_np = np.array(value).squeeze(-1)
        logprob_np = (
            logprob.cpu().numpy() if torch.is_tensor(logprob) else np.array(logprob)
        )
        self.value_buffer.append(value_np)
        self.log_prob_buffer.append(logprob_np)

    def update(self, next_obs=None, next_done=None):
        if len(self.obs_buffer) == 0:
            return {}

        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        actions = torch.LongTensor(np.array(self.action_buffer)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.reward_buffer)).to(self.device)
        dones = torch.FloatTensor(np.array(self.done_buffer)).to(self.device)
        masks = torch.FloatTensor(np.array(self.mask_buffer)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.value_buffer)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_prob_buffer)).to(
            self.device
        )

        with torch.no_grad():
            if next_obs is not None:
                next_obs_t = torch.FloatTensor(next_obs).to(self.device)
                next_value = self.model.get_value(next_obs_t).squeeze(-1)
                if next_done is not None:
                    next_value = next_value * (
                        1 - torch.FloatTensor(next_done).to(self.device)
                    )
            else:
                next_value = torch.zeros(rewards.shape[1], device=self.device)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        num_steps = rewards.shape[0]

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - (
                    torch.FloatTensor(next_done).to(self.device)
                    if next_done is not None
                    else 0
                )
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = old_values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_values * next_non_terminal
                - old_values[t]
            )
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
            )

        returns = advantages + old_values

        b_obs = obs.reshape(-1, self.obs_dim)
        b_actions = actions.reshape(-1)
        b_masks = masks.reshape(-1, self.action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        with autocast(enabled=self.use_amp):
            _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                b_obs, b_masks, b_actions
            )

            pg_loss = -(b_advantages * new_log_prob).mean()
            v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.mask_buffer.clear()
        self.value_buffer.clear()
        self.log_prob_buffer.clear()

        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
        }

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
