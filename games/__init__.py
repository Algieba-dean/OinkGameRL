"""Board game environments for reinforcement learning.

This package provides gymnasium-compatible environments for various board games
and card games, including Oink Games and traditional Chinese card games.

Quick Start:
    >>> from games import make_env, list_games
    >>> print(list_games())
    >>> env = make_env("doudizhu", render_mode="ansi")
    >>> obs, info = env.reset(seed=42)
"""

# Core components
from games.board_game import BoardGameEnv

# Import all game environments to trigger registration
from games.doudizhu.doudizhu_game_env import DoudizhuGameEnv
from games.game_agent import GameAgent
from games.guandan.guandan_game_env import GuandanGameEnv
from games.in_a_grove.in_a_grove_game_env import InAGroveGameEnv
from games.kobayakawa.kobayakawa_game_env import KobayakawaGameEnv
from games.mahjong.mahjong_game_env import MahjongGameEnv
from games.maskmen.maskmen_game_env import MaskmenGameEnv
from games.registry import get_game, list_games, make_env, register_game
from games.scout.scout_game_env import ScoutGameEnv
from games.startups.startups_game_env import StartupsGameEnv

__all__ = [
    # Core
    "BoardGameEnv",
    "GameAgent",
    # Registry
    "register_game",
    "get_game",
    "list_games",
    "make_env",
    # Oink Games
    "ScoutGameEnv",
    "KobayakawaGameEnv",
    "MaskmenGameEnv",
    "StartupsGameEnv",
    "InAGroveGameEnv",
    # Chinese Card Games
    "DoudizhuGameEnv",
    "GuandanGameEnv",
    "MahjongGameEnv",
]
