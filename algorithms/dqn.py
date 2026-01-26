"""DQN (Deep Q-Network) algorithm with action masking support."""

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from .base import BaseAlgorithm
from .registry import register_algorithm
from .replay_buffer import ReplayBuffer, ReplayBufferConfig


class QNetwork(nn.Module):
    """Q-Network: obs -> Q-values for all actions."""

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


@register_algorithm("dqn")
class DQN(BaseAlgorithm):
    """Deep Q-Network with action masking, target network, and epsilon-greedy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 256,
        min_buffer_size: int = 1000,
        target_update_freq: int = 1000,
        tau: float = 1.0,  # 1.0 = hard update, <1.0 = soft (polyak)
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        model: nn.Module | None = None,
        config: Any = None,
        **kwargs,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start

        self.q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()

        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

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

    def _compute_epsilon(self) -> float:
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

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

        batch_size = obs.shape[0]
        self.epsilon = self._compute_epsilon()

        with torch.no_grad():
            q_values = self.q_network(obs)

            if mask is not None:
                masked_q_values = q_values.clone()
                masked_q_values[mask == 0] = float("-inf")
            else:
                masked_q_values = q_values

            if deterministic:
                actions = masked_q_values.argmax(dim=-1)
            else:
                actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

                for i in range(batch_size):
                    if np.random.random() < self.epsilon:
                        if mask is not None:
                            valid_actions = torch.where(mask[i] == 1)[0]
                            if len(valid_actions) > 0:
                                random_idx = np.random.randint(len(valid_actions))
                                actions[i] = valid_actions[random_idx]
                            else:
                                actions[i] = np.random.randint(self.action_dim)
                        else:
                            actions[i] = np.random.randint(self.action_dim)
                    else:
                        actions[i] = masked_q_values[i].argmax()

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
            return {"loss": 0.0, "q_mean": 0.0, "epsilon": self.epsilon}

        batch = self.replay_buffer.sample(self.batch_size)

        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device).float()
        next_masks = batch["masks"].to(self.device)

        current_q_values = self.q_network(obs)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_obs)

            masked_next_q = next_q_values.clone()
            masked_next_q[next_masks == 0] = float("-inf")

            max_next_q = masked_next_q.max(dim=1)[0]

            max_next_q = torch.where(
                torch.isinf(max_next_q),
                torch.zeros_like(max_next_q),
                max_next_q,
            )

            # TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def _update_target_network(self):
        if self.tau == 1.0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            for target_param, param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, path: str):
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
