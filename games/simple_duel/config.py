from dataclasses import dataclass


@dataclass
class SimpleDuelConfig:
    """Configuration for SimpleDuel game."""

    name: str = "simple_duel"
    obs_dim: int = 160
    action_dim: int = 13
    default_model_name: str = "actor_critic"
    use_reward_shaping: bool = True

    # SimpleDuel-specific
    map_size: int = 12
    max_hp: int = 4
    max_energy: int = 7
    max_ammo: int = 6
    max_shield: int = 2
