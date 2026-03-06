"""Algorithm registry for extensible algorithm management."""

from typing import Any, Callable, Dict, Type
from .base import BaseAlgorithm


# Global algorithm registry
_ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {}


def register_algorithm(name: str) -> Callable:
    """
    Decorator to register an algorithm class.

    Usage:
        @register_algorithm("ppo")
        class PPO(BaseAlgorithm):
            ...
    """

    def decorator(cls: Type[BaseAlgorithm]) -> Type[BaseAlgorithm]:
        if name in _ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{name}' already registered")
        _ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm(name: str, **kwargs) -> BaseAlgorithm:
    """
    Create an algorithm instance by name.

    Args:
        name: Registered algorithm name
        **kwargs: Arguments to pass to algorithm constructor

    Returns:
        Instantiated algorithm
    """
    if name not in _ALGORITHM_REGISTRY:
        available = list(_ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorithm '{name}' not found. Available: {available}")
    return _ALGORITHM_REGISTRY[name](**kwargs)


def build_algorithm(
    name: str,
    *,
    config: Any,
    obs_dim: int,
    action_dim: int,
    model_name: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> BaseAlgorithm:
    """Create an algorithm with the right constructor contract.

    PPO/A2C consume `model_name`, while value-based algorithms currently do not.
    Routing construction through this helper keeps call sites consistent.
    """

    init_kwargs: dict[str, Any] = {
        "config": config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        **kwargs,
    }

    if name in {"ppo", "a2c"}:
        if model_name is None:
            raise ValueError(f"Algorithm '{name}' requires `model_name`")
        init_kwargs["model_name"] = model_name
    else:
        init_kwargs["device"] = device or str(getattr(config, "device", "cpu"))

    return get_algorithm(name, **init_kwargs)


def list_algorithms() -> list:
    """List all registered algorithm names."""
    return list(_ALGORITHM_REGISTRY.keys())


def get_algorithm_class(name: str) -> Type[BaseAlgorithm]:
    """Get algorithm class by name (without instantiating)."""
    if name not in _ALGORITHM_REGISTRY:
        available = list(_ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorithm '{name}' not found. Available: {available}")
    return _ALGORITHM_REGISTRY[name]
