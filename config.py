import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class GameConfig:
    """Base configuration for game-specific parameters."""

    name: str = "simple_duel"
    obs_dim: int = 160
    action_dim: int = 13
    action_names: List[str] = field(default_factory=list)

    # Game-specific reward shaping (optional)
    use_reward_shaping: bool = True

    def get_action_name(self, action_id: int) -> str:
        if 0 <= action_id < len(self.action_names):
            return self.action_names[action_id]
        return str(action_id)

    @classmethod
    def from_name(cls, name: str) -> "GameConfig":
        """Create GameConfig from game name."""
        configs = {
            "simple_duel": SimpleDuelConfig,
            "tictactoe": TicTacToeConfig,
            "connect4": Connect4Config,
            "reversi": ReversiConfig,
        }
        if name in configs:
            return configs[name]()
        # Try to get from Rust registry
        try:
            from envs import get_game_info

            info = get_game_info(name)
            return GameConfig(
                name=name,
                obs_dim=info.obs_dim,
                action_dim=info.action_dim,
                action_names=[str(i) for i in range(info.action_dim)],
                use_reward_shaping=False,
            )
        except Exception:
            raise ValueError(f"Unknown game: {name}")


@dataclass
class SimpleDuelConfig(GameConfig):
    """Configuration for SimpleDuel game."""

    name: str = "simple_duel"
    obs_dim: int = 160
    action_dim: int = 13
    action_names: List[str] = field(
        default_factory=lambda: [
            "Stay",
            "Up",
            "Down",
            "Left",
            "Right",
            "Attack",
            "Shoot",
            "Dodge",
            "Shield",
            "Dash",
            "AOE",
            "Heal",
            "Reload",
        ]
    )
    use_reward_shaping: bool = True

    # SimpleDuel-specific
    map_size: int = 12
    max_hp: int = 4
    max_energy: int = 7
    max_ammo: int = 6
    max_shield: int = 2


@dataclass
class TicTacToeConfig(GameConfig):
    """Configuration for TicTacToe game."""

    name: str = "tictactoe"
    obs_dim: int = 27
    action_dim: int = 9
    action_names: List[str] = field(default_factory=lambda: [str(i) for i in range(9)])
    use_reward_shaping: bool = False  # Simple win/lose rewards

    # TicTacToe-specific
    board_size: int = 3


@dataclass
class Connect4Config(GameConfig):
    """Configuration for Connect4 game."""

    name: str = "connect4"
    obs_dim: int = 126
    action_dim: int = 7
    action_names: List[str] = field(default_factory=lambda: [str(i) for i in range(7)])
    use_reward_shaping: bool = False


@dataclass
class ReversiConfig(GameConfig):
    """Configuration for Reversi game."""

    name: str = "reversi"
    obs_dim: int = 192
    action_dim: int = 64
    action_names: List[str] = field(default_factory=lambda: [str(i) for i in range(64)])
    use_reward_shaping: bool = False


@dataclass
class Config:
    # ============ 基础训练参数 ============
    total_timesteps: int = 10_000_000
    num_envs: int = 16
    num_steps: int = 2048

    # Derived fields (initialized in __post_init__ or by default)
    batch_size: int = field(init=False)
    minibatch_size: int = 4096
    num_minibatches: int = field(init=False)
    update_epochs: int = 4

    # ============ PPO 超参数 ============
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ============ 高级训练选项 ============
    target_kl: float = 0.02
    use_reward_shaping: bool = True
    use_opponent_pool: bool = False

    # ============ Model Selection ============
    model_name: str = "actor_critic"

    # ============ Performance Options ============
    use_amp: bool = False
    use_compile: bool = False
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    use_pinned_memory: bool = True

    # ============ 评估与可视化 ============
    eval_interval: int = 25
    save_interval: int = 100
    eval_opponent: str = "self"

    # Game configuration
    game: str = "simple_duel"
    game_config: GameConfig = field(default_factory=lambda: SimpleDuelConfig())

    # Non-init fields
    device: torch.device = field(init=False)
    train_steps: int = field(init=False)

    def __post_init__(self):
        # Auto-create game_config if game name changed
        if self.game != self.game_config.name:
            self.game_config = GameConfig.from_name(self.game)

        # Calculate derived values
        self.batch_size = 2 * self.num_envs * self.num_steps
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.train_steps = self.total_timesteps // self.batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")


class FastConfig(Config):
    """快速训练配置 - 用于测试"""

    def __init__(self):
        super().__init__(total_timesteps=1_000_000, num_steps=1024)
        # Recalculate derived values that depend on init args if necessary
        # Since we passed them to super().__init__, __post_init__ handled them correctly!
        # But wait, FastConfig logic in original code was:
        # super().__init__() (which set defaults)
        # then overwrite attributes
        # then recalculate train_steps

        # With dataclass, we can pass args to __init__
        pass


class LongConfig(Config):
    """长时间训练配置 - 追求最强模型"""

    def __init__(self):
        super().__init__(
            total_timesteps=50_000_000,
            num_envs=32,
            minibatch_size=8192,
            learning_rate=1e-4,
            ent_coef=0.005,
            use_opponent_pool=True,
        )
