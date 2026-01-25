import torch
from dataclasses import dataclass, field


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

    # Non-init fields
    device: torch.device = field(init=False)
    train_steps: int = field(init=False)

    def __post_init__(self):
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
