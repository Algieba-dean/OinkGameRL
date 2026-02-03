"""Game registry for dynamic game environment loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.board_game import BoardGameEnv

_GAME_REGISTRY: dict[str, type[BoardGameEnv]] = {}


def register_game(name: str):
    """Decorator to register a game environment class.

    Args:
        name: Unique identifier for the game

    Example:
        @register_game("scout")
        class ScoutGameEnv(BoardGameEnv):
            ...
    """

    def decorator(cls: type[BoardGameEnv]) -> type[BoardGameEnv]:
        if name in _GAME_REGISTRY:
            raise ValueError(f"Game '{name}' is already registered")
        _GAME_REGISTRY[name] = cls
        return cls

    return decorator


def get_game(name: str) -> type[BoardGameEnv]:
    """Get a game environment class by name.

    Args:
        name: The registered name of the game

    Returns:
        The game environment class

    Raises:
        KeyError: If game is not registered
    """
    if name not in _GAME_REGISTRY:
        raise KeyError(
            f"Game '{name}' not found. Available games: {list(_GAME_REGISTRY.keys())}"
        )
    return _GAME_REGISTRY[name]


def list_games() -> list[str]:
    """List all registered game names.

    Returns:
        List of registered game names
    """
    return list(_GAME_REGISTRY.keys())


def make_env(name: str, **kwargs) -> BoardGameEnv:
    """Create a game environment instance by name.

    Args:
        name: The registered name of the game
        **kwargs: Arguments to pass to the environment constructor

    Returns:
        An instance of the game environment
    """
    game_cls = get_game(name)
    return game_cls(**kwargs)
