"""Pre-allocated rollout buffer for efficient PPO training."""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Generator


@dataclass
class RolloutBufferConfig:
    """Configuration for rollout buffer."""

    num_steps: int
    num_envs: int
    obs_dim: int
    action_dim: int
    device: str = "cpu"
    use_pinned_memory: bool = True


class RolloutBuffer:
    """
    Pre-allocated rollout buffer with pinned memory support.

    Eliminates per-step allocations by pre-allocating all tensors upfront.
    Supports pinned memory for faster CPU->GPU transfers.
    """

    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        self.num_steps = config.num_steps
        self.num_envs = config.num_envs
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.device = config.device

        # Determine if we should use pinned memory
        self.use_pinned = config.use_pinned_memory and torch.cuda.is_available()

        # Pre-allocate all buffers
        self._allocate_buffers()

        # Current position in buffer
        self.pos = 0
        self.full = False

    def _allocate_buffers(self):
        """Allocate all buffers upfront."""
        pin = self.use_pinned

        # Observations: [num_steps, num_envs, obs_dim]
        self.obs = torch.zeros(
            (self.num_steps, self.num_envs, self.obs_dim),
            dtype=torch.float32,
            pin_memory=pin,
        )

        # Next observations (for reward shaping): [num_steps, num_envs, obs_dim]
        self.next_obs = torch.zeros(
            (self.num_steps, self.num_envs, self.obs_dim),
            dtype=torch.float32,
            pin_memory=pin,
        )

        # Action masks: [num_steps, num_envs, action_dim]
        self.masks = torch.zeros(
            (self.num_steps, self.num_envs, self.action_dim),
            dtype=torch.float32,
            pin_memory=pin,
        )

        # Actions: [num_steps, num_envs]
        self.actions = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.long, pin_memory=pin
        )

        # Log probabilities: [num_steps, num_envs]
        self.logprobs = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32, pin_memory=pin
        )

        # Rewards: [num_steps, num_envs]
        self.rewards = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32, pin_memory=pin
        )

        # Values: [num_steps, num_envs]
        self.values = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32, pin_memory=pin
        )

        # Dones: [num_steps, num_envs]
        self.dones = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.bool, pin_memory=pin
        )

        # Advantages and returns (computed later): [num_steps, num_envs]
        self.advantages = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32, pin_memory=pin
        )
        self.returns = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32, pin_memory=pin
        )

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        logprob: torch.Tensor,
        mask: torch.Tensor,
        next_obs: Optional[torch.Tensor] = None,
    ):
        """
        Add a transition to the buffer.

        All inputs should be [num_envs, ...] shaped tensors.
        """
        self.obs[self.pos].copy_(obs)
        self.actions[self.pos].copy_(action)
        self.rewards[self.pos].copy_(reward)
        self.dones[self.pos].copy_(done)
        self.values[self.pos].copy_(value)
        self.logprobs[self.pos].copy_(logprob)
        self.masks[self.pos].copy_(mask)

        if next_obs is not None:
            self.next_obs[self.pos].copy_(next_obs)

        self.pos += 1
        if self.pos == self.num_steps:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE advantages and returns.

        Args:
            last_value: Value estimate for the state after the last step [num_envs]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        last_gae_lam = 0

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_value
            else:
                next_values = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )
            self.advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def get_samples(self, batch_size: int) -> Generator:
        """
        Generate random minibatches for training.

        Yields flattened tensors of shape [batch_size, ...].
        """
        total_size = self.num_steps * self.num_envs
        indices = np.random.permutation(total_size)

        # Flatten all buffers
        flat_obs = self.obs.reshape(-1, self.obs_dim)
        flat_masks = self.masks.reshape(-1, self.action_dim)
        flat_actions = self.actions.reshape(-1)
        flat_logprobs = self.logprobs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                "obs": flat_obs[batch_indices],
                "masks": flat_masks[batch_indices],
                "actions": flat_actions[batch_indices],
                "logprobs": flat_logprobs[batch_indices],
                "advantages": flat_advantages[batch_indices],
                "returns": flat_returns[batch_indices],
                "values": flat_values[batch_indices],
            }

    def to_device(self, device: str, non_blocking: bool = True):
        """Move all buffers to specified device."""
        self.obs = self.obs.to(device, non_blocking=non_blocking)
        self.next_obs = self.next_obs.to(device, non_blocking=non_blocking)
        self.masks = self.masks.to(device, non_blocking=non_blocking)
        self.actions = self.actions.to(device, non_blocking=non_blocking)
        self.logprobs = self.logprobs.to(device, non_blocking=non_blocking)
        self.rewards = self.rewards.to(device, non_blocking=non_blocking)
        self.values = self.values.to(device, non_blocking=non_blocking)
        self.dones = self.dones.to(device, non_blocking=non_blocking)
        self.advantages = self.advantages.to(device, non_blocking=non_blocking)
        self.returns = self.returns.to(device, non_blocking=non_blocking)
        self.device = device

    def reset(self):
        """Reset buffer position for next rollout."""
        self.pos = 0
        self.full = False
