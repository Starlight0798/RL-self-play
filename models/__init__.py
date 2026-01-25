"""Models package."""

from .registry import register_model, get_model, list_models, get_model_class
import model  # Register models

__all__ = ["register_model", "get_model", "list_models", "get_model_class"]
