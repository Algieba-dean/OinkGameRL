"""Tests for game registry system."""

import pytest

from games.board_game import BoardGameEnv
from games.registry import (
    _GAME_REGISTRY,
    get_game,
    list_games,
    make_env,
    register_game,
)


class TestRegisterGame:
    """Test register_game decorator."""

    def test_register_new_game(self):
        """Register a new game successfully."""
        original_registry = _GAME_REGISTRY.copy()

        @register_game("test_game_1")
        class TestGame1(BoardGameEnv):
            pass

        assert "test_game_1" in _GAME_REGISTRY
        assert _GAME_REGISTRY["test_game_1"] is TestGame1

        _GAME_REGISTRY.clear()
        _GAME_REGISTRY.update(original_registry)

    def test_register_duplicate_raises_error(self):
        """Registering duplicate game name raises ValueError."""
        original_registry = _GAME_REGISTRY.copy()

        @register_game("test_game_dup")
        class TestGameDup1(BoardGameEnv):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_game("test_game_dup")
            class TestGameDup2(BoardGameEnv):
                pass

        _GAME_REGISTRY.clear()
        _GAME_REGISTRY.update(original_registry)


class TestGetGame:
    """Test get_game function."""

    def test_get_registered_game(self):
        """Get a registered game class."""
        original_registry = _GAME_REGISTRY.copy()

        @register_game("test_get_game")
        class TestGetGameEnv(BoardGameEnv):
            pass

        result = get_game("test_get_game")
        assert result is TestGetGameEnv

        _GAME_REGISTRY.clear()
        _GAME_REGISTRY.update(original_registry)

    def test_get_unregistered_game_raises_error(self):
        """Getting unregistered game raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_game("nonexistent_game")


class TestListGames:
    """Test list_games function."""

    def test_list_games_returns_registered_names(self):
        """List all registered game names."""
        original_registry = _GAME_REGISTRY.copy()
        _GAME_REGISTRY.clear()

        @register_game("game_a")
        class GameA(BoardGameEnv):
            pass

        @register_game("game_b")
        class GameB(BoardGameEnv):
            pass

        result = list_games()
        assert "game_a" in result
        assert "game_b" in result

        _GAME_REGISTRY.clear()
        _GAME_REGISTRY.update(original_registry)


class TestMakeEnv:
    """Test make_env function."""

    def test_make_env_creates_instance(self):
        """make_env creates environment instance."""
        from games.scout.scout_game_env import ScoutGameEnv

        if "scout" not in _GAME_REGISTRY:
            _GAME_REGISTRY["scout"] = ScoutGameEnv

        env = make_env("scout", player_num=4)
        assert isinstance(env, ScoutGameEnv)
        assert env.num_players == 4

    def test_make_env_with_kwargs(self):
        """make_env passes kwargs to constructor."""
        from games.scout.scout_game_env import ScoutGameEnv

        if "scout" not in _GAME_REGISTRY:
            _GAME_REGISTRY["scout"] = ScoutGameEnv

        env = make_env("scout", player_num=3, render_mode="ansi")
        assert env.num_players == 3
        assert env.render_mode == "ansi"
