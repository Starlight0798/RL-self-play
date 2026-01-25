"""Environment registry and factory for RL self-play games."""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

# Import Rust bindings
import high_perf_env


@dataclass
class GameInfo:
    """Information about a registered game."""

    name: str
    obs_dim: int
    action_dim: int
    description: str = ""


# Game registry
_GAME_REGISTRY: Dict[str, GameInfo] = {}


def register_game(
    name: str, obs_dim: int, action_dim: int, description: str = ""
) -> None:
    """Register a game in the Python registry."""
    _GAME_REGISTRY[name] = GameInfo(name, obs_dim, action_dim, description)


def get_game_info(name: str) -> GameInfo:
    """Get information about a registered game."""
    if name not in _GAME_REGISTRY:
        # Try to get from Rust side
        try:
            obs_dim, action_dim = high_perf_env.get_game_info(name)
            register_game(name, obs_dim, action_dim)
        except Exception as e:
            raise ValueError(f"Unknown game: {name}. Available: {list_games()}") from e
    return _GAME_REGISTRY[name]


def list_games() -> list:
    """List all available games."""
    # Combine Python and Rust registries
    rust_games = high_perf_env.list_games()
    return list(set(list(_GAME_REGISTRY.keys()) + rust_games))


def create_env(game_name: str, num_envs: int, zero_copy: bool = True):
    """Create a vectorized environment for the specified game.

    Args:
        game_name: Name of the game ('simple_duel', 'tictactoe', etc.)
        num_envs: Number of parallel environments
        zero_copy: If True, use zero-copy environment (faster)

    Returns:
        VectorizedEnvGeneric instance
    """
    return high_perf_env.create_env(game_name, num_envs)


# Auto-register known games with descriptions
def _init_registry():
    """Initialize registry with known games."""
    try:
        for game_name in high_perf_env.list_games():
            obs_dim, action_dim = high_perf_env.get_game_info(game_name)
            descriptions = {
                "simple_duel": "12x12 tactical combat with 13 actions",
                "tictactoe": "3x3 board game with 9 actions",
            }
            register_game(
                game_name, obs_dim, action_dim, descriptions.get(game_name, "")
            )
    except Exception:
        pass  # Rust module not built yet


_init_registry()
