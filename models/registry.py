"""Model registry for extensible model management."""

from typing import Dict, Type, Callable, Any, Optional
import torch.nn as nn


# Global model registry
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model class.

    Usage:
        @register_model("actor_critic")
        class ActorCritic(nn.Module):
            ...
    """

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Create a model instance by name.

    Args:
        name: Registered model name
        **kwargs: Arguments to pass to model constructor

    Returns:
        Instantiated model
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())


def get_model_class(name: str) -> Type[nn.Module]:
    """Get model class by name (without instantiating)."""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    return _MODEL_REGISTRY[name]
