from dataclasses import dataclass


@dataclass
class TicTacToeConfig:
    """Configuration for TicTacToe game."""

    name: str = "tictactoe"
    obs_dim: int = 27
    action_dim: int = 9
    default_model_name: str = "simple_mlp"
    use_reward_shaping: bool = False

    # TicTacToe-specific
    board_size: int = 3
