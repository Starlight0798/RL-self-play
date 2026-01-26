"""Discrete SAC (Soft Actor-Critic) algorithm with action masking support.

Reference: "Soft Actor-Critic for Discrete Action Settings" (Christodoulou, 2019)
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

from .base import BaseAlgorithm
from .registry import register_algorithm
from .replay_buffer import ReplayBuffer, ReplayBufferConfig


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(obs)

        if mask is not None:
            logits = logits + (mask - 1.0) * 1e10

        action_probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        return action_probs, log_probs


class DiscreteCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


@register_algorithm("sac")
class DiscreteSAC(BaseAlgorithm):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        min_buffer_size: int = 1000,
        target_entropy_ratio: float = 0.98,
        auto_alpha: bool = True,
        initial_alpha: float = 0.2,
        model: nn.Module | None = None,
        config: Any = None,
        **kwargs,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.auto_alpha = auto_alpha

        self.actor = DiscreteActor(obs_dim, action_dim, hidden_dim).to(device)

        self.q1 = DiscreteCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.q2 = DiscreteCritic(obs_dim, action_dim, hidden_dim).to(device)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q1_target.eval()
        self.q2_target.eval()

        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(
            np.log(initial_alpha),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # target_entropy = -ratio * log(1/|A|) = ratio * log(|A|)
        self.target_entropy = -target_entropy_ratio * np.log(1.0 / action_dim)

        buffer_config = ReplayBufferConfig(
            max_size=buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device="cpu",
            use_pinned_memory=torch.cuda.is_available(),
        )
        self.replay_buffer = ReplayBuffer(buffer_config)

        self.training_step = 0
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def get_action(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.FloatTensor(mask)

        obs = obs.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            action_probs, _ = self.actor(obs, mask)

            if deterministic:
                actions = action_probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()

        return actions, {}

    def store_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        mask: torch.Tensor,
        info: dict[str, Any],
        next_obs: torch.Tensor | None = None,
        next_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if next_obs is None:
            return

        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.FloatTensor(next_obs)
        if isinstance(action, np.ndarray):
            action = torch.LongTensor(action)
        if isinstance(reward, np.ndarray):
            reward = torch.FloatTensor(reward)
        if isinstance(done, np.ndarray):
            done = torch.BoolTensor(done)
        if isinstance(mask, np.ndarray):
            mask = torch.FloatTensor(mask)
        if next_mask is not None and isinstance(next_mask, np.ndarray):
            next_mask = torch.FloatTensor(next_mask)

        obs = obs.cpu()
        next_obs = next_obs.cpu()
        action = action.cpu()
        reward = reward.cpu()
        done = done.cpu()
        mask = mask.cpu()

        batch_size = obs.shape[0]
        for i in range(batch_size):
            action_val = (
                int(action[i].item()) if action.dim() > 0 else int(action.item())
            )
            done_val = bool(done[i].item()) if done.dim() > 0 else bool(done.item())
            self.replay_buffer.add(
                obs=obs[i],
                action=action_val,
                reward=reward[i].item() if reward.dim() > 0 else reward.item(),
                next_obs=next_obs[i],
                done=done_val,
                mask=next_mask[i] if next_mask is not None else mask[i],
            )

        self.total_steps += batch_size

    def update(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        if len(self.replay_buffer) < self.min_buffer_size:
            return {
                "actor_loss": 0.0,
                "q1_loss": 0.0,
                "q2_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": self.alpha.item(),
                "entropy": 0.0,
            }

        batch = self.replay_buffer.sample(self.batch_size)

        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device).float()
        next_masks = batch["masks"].to(self.device)

        with torch.no_grad():
            next_action_probs, next_log_probs = self.actor(next_obs, next_masks)

            next_q1 = self.q1_target(next_obs)
            next_q2 = self.q2_target(next_obs)
            next_q_min = torch.min(next_q1, next_q2)

            # V(s') = sum_a' pi(a'|s') * (Q(s',a') - alpha * log(pi(a'|s')))
            next_v = (
                next_action_probs * (next_q_min - self.alpha * next_log_probs)
            ).sum(dim=-1)

            target_q = rewards + self.gamma * next_v * (1 - dones)

        current_q1 = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=10.0)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=10.0)
        self.q2_optimizer.step()

        action_probs, log_probs = self.actor(obs, None)

        with torch.no_grad():
            q1_values = self.q1(obs)
            q2_values = self.q2(obs)
            q_min = torch.min(q1_values, q2_values)

        # actor_loss = sum_a pi(a|s) * (alpha * log(pi(a|s)) - Q(s,a))
        actor_loss = (
            (action_probs * (self.alpha.detach() * log_probs - q_min))
            .sum(dim=-1)
            .mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        if self.auto_alpha:
            with torch.no_grad():
                action_probs_detached, log_probs_detached = self.actor(obs, None)
                entropy = (
                    -(action_probs_detached * log_probs_detached).sum(dim=-1).mean()
                )

            alpha_loss = self.log_alpha * (entropy - self.target_entropy)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
            entropy = -(action_probs * log_probs).sum(dim=-1).mean()

        self._soft_update_target_networks()

        self.training_step += 1

        return {
            "actor_loss": actor_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0.0,
            "alpha": self.alpha.item(),
            "entropy": entropy.item(),
        }

    def _soft_update_target_networks(self):
        for target_param, param in zip(
            self.q1_target.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.q2_target.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
            "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "training_step": self.training_step,
            "total_steps": self.total_steps,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])

        if "log_alpha" in checkpoint:
            self.log_alpha = torch.tensor(
                checkpoint["log_alpha"],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.alpha_optimizer.param_groups[0]["lr"]
            )
            if "alpha_optimizer_state_dict" in checkpoint:
                self.alpha_optimizer.load_state_dict(
                    checkpoint["alpha_optimizer_state_dict"]
                )

        self.training_step = checkpoint.get("training_step", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
