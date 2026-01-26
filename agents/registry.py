"""Registry for rule-based agents across different games."""

import torch
import numpy as np
from typing import Dict, Type, Callable, Any, Optional, List, Tuple


# Global rule agent registry
_RULE_AGENT_REGISTRY: Dict[str, Type[Any]] = {}


def register_rule_agent(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a rule agent class.

    Usage:
        @register_rule_agent("simple_duel")
        class SimpleDuelRuleAgent:
            ...
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        if name in _RULE_AGENT_REGISTRY:
            raise ValueError(f"Rule agent '{name}' already registered")
        _RULE_AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_rule_agent(name: str, device: str = "cpu", **kwargs: Any) -> Any:
    """
    Create a rule agent instance by name.

    Args:
        name: Registered game name
        device: Device to put actions on
        **kwargs: Additional arguments for the agent constructor

    Returns:
        Instantiated rule agent
    """
    if name not in _RULE_AGENT_REGISTRY:
        available = list(_RULE_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Rule agent for game '{name}' not found. Available: {available}"
        )
    return _RULE_AGENT_REGISTRY[name](device=device, **kwargs)


def list_rule_agents() -> list[str]:
    """List all registered rule agent names."""
    return list(_RULE_AGENT_REGISTRY.keys())


class RandomRuleAgent:
    """Generic random agent that picks a valid action from the mask."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def get_action(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Pick a random valid action.

        Args:
            obs: Observation tensor [B, OBS_DIM]
            mask: Action mask [B, ACTION_DIM]
            deterministic: Ignored for random agent

        Returns:
            (actions, info_dict)
        """
        batch_size = obs.shape[0]
        if mask is not None:
            mask_np = mask.cpu().numpy()
            actions = []
            for i in range(batch_size):
                valid_indices = np.where(mask_np[i] > 0.5)[0]
                if len(valid_indices) > 0:
                    actions.append(np.random.choice(valid_indices))
                else:
                    # Fallback to action 0 if no valid actions (shouldn't happen)
                    actions.append(0)
            return torch.LongTensor(actions).to(self.device), {}
        else:
            # If no mask is provided, we can't safely pick a random action without knowing ACTION_DIM
            raise ValueError(
                "RandomRuleAgent requires an action mask to select valid actions."
            )


# Register placeholder random agents for other games
@register_rule_agent("tictactoe")
class TicTacToeRandomAgent(RandomRuleAgent):
    pass


@register_rule_agent("connect4")
class Connect4RandomAgent(RandomRuleAgent):
    pass


@register_rule_agent("reversi")
class ReversiRandomAgent(RandomRuleAgent):
    pass
