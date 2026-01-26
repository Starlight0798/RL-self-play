import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from models.registry import register_model, get_model


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    """残差块，帮助训练更深的网络"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim), std=0.1)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x + residual)
        return x


class SpatialEncoder(nn.Module):
    """空间特征编码器 - 用于处理 12x12 地形网格"""

    def __init__(self, grid_size=12, grid_channels=1, output_dim=64):
        super().__init__()
        self.grid_size = grid_size

        # 使用 1D 卷积处理展平的网格 (更轻量)
        # 或者使用简单的 MLP 处理
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(grid_size * grid_size, 128)),
            nn.GELU(),
            layer_init(nn.Linear(128, output_dim)),
            nn.GELU(),
        )

    def forward(self, grid_flat):
        # grid_flat: [B, 144]
        return self.encoder(grid_flat)


@register_model("actor_critic")
class ActorCritic(nn.Module):
    """
    改进的 Actor-Critic 网络:
    1. 分离的 Actor 和 Critic 网络 (减少干扰)
    2. 空间特征编码器
    3. 残差连接
    4. Layer Normalization
    5. GELU 激活函数
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        state_dim: int = 16,
        grid_size: int = 12,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 状态特征维度: 前 state_dim 维是数值特征
        self.state_dim = state_dim
        self.grid_size = grid_size
        self.grid_dim = obs_dim - self.state_dim  # 144

        # 空间编码器
        self.spatial_encoder = SpatialEncoder(grid_size=grid_size, output_dim=64)

        # 状态特征编码器
        self.state_encoder = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.GELU(),
        )

        # 融合特征维度: 64 (state) + 64 (spatial) = 128
        fusion_dim = 128

        # =============== Actor 网络 (独立) ===============
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.actor_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        # =============== Critic 网络 (独立) ===============
        self.critic_backbone = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _encode_obs(self, x):
        """编码观测"""
        # 分离状态特征和空间特征
        state_features = x[:, : self.state_dim]
        grid_features = x[:, self.state_dim :]

        # 编码
        state_encoded = self.state_encoder(state_features)
        spatial_encoded = self.spatial_encoder(grid_features)

        # 融合
        fused = torch.cat([state_encoded, spatial_encoded], dim=-1)  # [B, 128]
        return fused

    def get_value(self, x):
        fused = self._encode_obs(x)
        hidden = self.critic_backbone(fused)
        return self.critic_head(hidden)

    def get_action_and_value(self, x, action_mask=None, action=None):
        fused = self._encode_obs(x)

        # Actor
        actor_hidden = self.actor_backbone(fused)
        logits = self.actor_head(actor_hidden)

        # Apply Action Masking
        if action_mask is not None:
            HUGE_NEG = -1e8
            logits = logits + (action_mask - 1.0) * (-HUGE_NEG)

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        # Critic
        critic_hidden = self.critic_backbone(fused)
        value = self.critic_head(critic_hidden)

        return action, probs.log_prob(action), probs.entropy(), value


@register_model("dueling_actor_critic")
class DuelingActorCritic(nn.Module):
    """
    Dueling 架构的 Actor-Critic:
    - Actor 使用 Advantage 分解
    - 更稳定的价值估计
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        state_dim: int = 16,
        grid_size: int = 12,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.grid_size = grid_size
        self.grid_dim = obs_dim - self.state_dim

        # 空间编码器
        self.spatial_encoder = SpatialEncoder(grid_size=grid_size, output_dim=64)

        # 状态特征编码器
        self.state_encoder = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.GELU(),
        )

        fusion_dim = 128

        # 共享特征提取
        self.shared_backbone = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
        )

        # Actor 分支
        self.actor_branch = nn.Sequential(
            ResidualBlock(hidden_dim),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        # Critic V(s) 分支
        self.value_branch = nn.Sequential(
            ResidualBlock(hidden_dim),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _encode_obs(self, x):
        state_features = x[:, : self.state_dim]
        grid_features = x[:, self.state_dim :]
        state_encoded = self.state_encoder(state_features)
        spatial_encoded = self.spatial_encoder(grid_features)
        return torch.cat([state_encoded, spatial_encoded], dim=-1)

    def get_value(self, x):
        fused = self._encode_obs(x)
        shared = self.shared_backbone(fused)
        return self.value_branch(shared)

    def get_action_and_value(self, x, action_mask=None, action=None):
        fused = self._encode_obs(x)
        shared = self.shared_backbone(fused)

        logits = self.actor_branch(shared)

        if action_mask is not None:
            HUGE_NEG = -1e8
            logits = logits + (action_mask - 1.0) * (-HUGE_NEG)

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.value_branch(shared)

        return action, probs.log_prob(action), probs.entropy(), value


@register_model("cnn")
class CNNActorCritic(nn.Module):
    """CNN Actor-Critic for board games (Connect4, Reversi). Expects obs as flattened 3-channel board."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        board_shape: tuple[int, int] = (6, 7),
        hidden_dim: int = 128,
        num_channels: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.board_shape = board_shape
        self.num_channels = num_channels
        rows, cols = board_shape

        expected_obs_dim = rows * cols * num_channels
        if obs_dim != expected_obs_dim:
            raise ValueError(
                f"obs_dim ({obs_dim}) != rows*cols*channels ({expected_obs_dim})"
            )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_output_size = 64 * rows * cols

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(conv_output_size, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        rows, cols = self.board_shape
        x = obs.view(batch_size, self.num_channels, rows, cols)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        return self.backbone(x)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._encode_obs(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._encode_obs(obs)
        logits = self.actor(features)

        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e8

        probs = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = masked_logits.argmax(dim=-1)
            else:
                action = probs.sample()

        assert action is not None
        value = self.critic(features)

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self._encode_obs(obs)
        return self.critic(features).squeeze(-1)


@register_model("simple_mlp")
class SimpleMLP(nn.Module):
    """Simple MLP actor-critic for small observation spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.actor(features)

        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e8

        probs = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = masked_logits.argmax(dim=-1)
            else:
                action = probs.sample()

        assert action is not None
        value = self.critic(features)

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.critic(features).squeeze(-1)


def create_model_for_game(
    game_name: str,
    obs_dim: int,
    action_dim: int,
    model_name: str | None = None,
) -> nn.Module:
    """Create an appropriate model for the given game.

    Args:
        game_name: Name of the game
        obs_dim: Observation dimension
        action_dim: Action dimension
        model_name: Optional specific model name. If None, auto-selects.

    Returns:
        Instantiated model
    """
    if model_name is None:
        if game_name == "simple_duel":
            model_name = "actor_critic"
        elif game_name == "tictactoe":
            model_name = "simple_mlp"
        else:
            model_name = "simple_mlp"

    kwargs = {}
    if game_name == "simple_duel":
        kwargs["state_dim"] = 16
        kwargs["grid_size"] = 12

    return get_model(model_name, obs_dim=obs_dim, action_dim=action_dim, **kwargs)
