import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from models.registry import register_model, get_model
from games.registry import get_game


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

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 状态特征维度: 前 16 维是数值特征
        self.state_dim = 16
        self.grid_dim = obs_dim - self.state_dim  # 144

        # 空间编码器
        self.spatial_encoder = SpatialEncoder(grid_size=12, output_dim=64)

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
        state_features = x[:, : self.state_dim]  # [B, 16]
        grid_features = x[:, self.state_dim :]  # [B, 144]

        # 编码
        state_encoded = self.state_encoder(state_features)  # [B, 64]
        spatial_encoded = self.spatial_encoder(grid_features)  # [B, 64]

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

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = 16
        self.grid_dim = obs_dim - self.state_dim

        # 空间编码器
        self.spatial_encoder = SpatialEncoder(grid_size=12, output_dim=64)

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
        model_name: Optional specific model name. If None, auto-selects from game config.

    Returns:
        Instantiated model
    """
    if model_name is None:
        try:
            game = get_game(game_name)
            # Use getattr to avoid crash if default_model_name is missing
            model_name = getattr(game.config, "default_model_name", "simple_mlp")
        except (ImportError, AttributeError):
            model_name = "simple_mlp"

    assert model_name is not None
    return get_model(model_name, obs_dim=obs_dim, action_dim=action_dim)
