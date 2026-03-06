"""Environment registry and factory for RL self-play games."""

from dataclasses import dataclass
from typing import Dict

try:
    import high_perf_env
except ModuleNotFoundError:
    high_perf_env = None


@dataclass
class GameInfo:
    """Information about a registered game."""

    name: str
    obs_dim: int
    action_dim: int
    description: str = ""


# Game registry
_GAME_REGISTRY: Dict[str, GameInfo] = {}


def has_native_backend() -> bool:
    """Whether the Rust extension module is available."""
    return high_perf_env is not None


def _require_backend():
    if high_perf_env is None:
        raise RuntimeError(
            "Rust extension `high_perf_env` is not built. "
            "Run `maturin develop --release` first."
        )
    return high_perf_env


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
            backend = _require_backend()
            obs_dim, action_dim = backend.get_game_info(name)
            register_game(name, obs_dim, action_dim)
        except Exception as e:
            raise ValueError(f"Unknown game: {name}. Available: {list_games()}") from e
    return _GAME_REGISTRY[name]


def list_games() -> list[str]:
    """List all available games."""
    rust_games: list[str] = []
    if has_native_backend():
        rust_games = _require_backend().list_games()
    return sorted(set(_GAME_REGISTRY.keys()) | set(rust_games))


def create_env(game_name: str, num_envs: int, zero_copy: bool = True):
    """Create a vectorized environment for the specified game.

    Args:
        game_name: Name of the game ('simple_duel', 'tictactoe', etc.)
        num_envs: Number of parallel environments
        zero_copy: If True, use zero-copy environment (faster)

    Returns:
        Native vectorized environment instance
    """
    backend = _require_backend()

    if game_name == "simple_duel":
        if zero_copy:
            return backend.VectorizedEnvZeroCopy(num_envs)
        return backend.VectorizedEnv(num_envs)

    return backend.create_env(game_name, num_envs)


# Auto-register known games with descriptions
def _init_registry():
    """Initialize registry with known games."""
    if not has_native_backend():
        return

    try:
        backend = _require_backend()
        for game_name in backend.list_games():
            obs_dim, action_dim = backend.get_game_info(game_name)
            descriptions = {
                "simple_duel": "12x12 tactical combat with 13 actions",
                "tictactoe": "3x3 board game with 9 actions",
            }
            register_game(
                game_name, obs_dim, action_dim, descriptions.get(game_name, "")
            )
    except Exception:
        return


_init_registry()
