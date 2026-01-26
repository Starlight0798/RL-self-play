import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class ReplayBufferConfig:
    max_size: int
    obs_dim: int
    action_dim: int
    device: str = "cpu"
    use_pinned_memory: bool = True


class ReplayBuffer:
    config: ReplayBufferConfig
    max_size: int
    obs_dim: int
    action_dim: int
    device: str
    use_pinned: bool
    pos: int
    size: int
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    masks: torch.Tensor

    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.max_size = config.max_size
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.device = config.device

        self.use_pinned = config.use_pinned_memory and torch.cuda.is_available()
        self._allocate_buffers()

        self.pos = 0
        self.size = 0

    def _allocate_buffers(self):
        pin = self.use_pinned

        self.obs = torch.zeros(
            (self.max_size, self.obs_dim),
            dtype=torch.float32,
            pin_memory=pin,
        )
        self.next_obs = torch.zeros(
            (self.max_size, self.obs_dim),
            dtype=torch.float32,
            pin_memory=pin,
        )
        self.actions = torch.zeros((self.max_size,), dtype=torch.long, pin_memory=pin)
        self.rewards = torch.zeros(
            (self.max_size,), dtype=torch.float32, pin_memory=pin
        )
        self.dones = torch.zeros((self.max_size,), dtype=torch.bool, pin_memory=pin)
        self.masks = torch.zeros(
            (self.max_size, self.action_dim),
            dtype=torch.float32,
            pin_memory=pin,
        )

    def add(
        self,
        obs: torch.Tensor,
        action: int | torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
        mask: torch.Tensor,
    ):
        self.obs[self.pos].copy_(obs)
        self.next_obs[self.pos].copy_(next_obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.masks[self.pos].copy_(mask)

        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": self.obs[indices],
            "next_obs": self.next_obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "masks": self.masks[indices],
        }

    def __len__(self) -> int:
        return self.size

    def to_device(self, device: str, non_blocking: bool = True):
        self.obs = self.obs.to(device, non_blocking=non_blocking)
        self.next_obs = self.next_obs.to(device, non_blocking=non_blocking)
        self.actions = self.actions.to(device, non_blocking=non_blocking)
        self.rewards = self.rewards.to(device, non_blocking=non_blocking)
        self.dones = self.dones.to(device, non_blocking=non_blocking)
        self.masks = self.masks.to(device, non_blocking=non_blocking)
        self.device = device
