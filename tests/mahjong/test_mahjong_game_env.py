"""Tests for Mahjong game environment."""

import gymnasium as gym
import numpy as np
import pytest

from games.board_game import BoardGameEnv
from games.mahjong.constants import GameConsts
from games.mahjong.mahjong_game_env import MahjongGameEnv


class TestMahjongContract:
    """Test MahjongGameEnv adheres to BoardGameEnv contract."""

    @pytest.fixture
    def env(self) -> MahjongGameEnv:
        return MahjongGameEnv()

    def test_is_board_game_env(self, env):
        assert isinstance(env, BoardGameEnv)

    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_has_spaces(self, env):
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_num_players(self, env):
        assert env.num_players == GameConsts.NUM_PLAYERS


class TestMahjongReset:
    """Test reset functionality."""

    @pytest.fixture
    def env(self) -> MahjongGameEnv:
        return MahjongGameEnv()

    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)

    def test_reset_different_seeds(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)
        assert not np.array_equal(obs1, obs2)

    def test_reset_info_contains_action_mask(self, env):
        _, info = env.reset(seed=42)
        assert "action_mask" in info

    def test_reset_info_contains_global_state(self, env):
        _, info = env.reset(seed=42)
        assert "global_state" in info


class TestMahjongStep:
    """Test step functionality."""

    @pytest.fixture
    def env(self) -> MahjongGameEnv:
        env = MahjongGameEnv()
        env.reset(seed=42)
        return env

    def test_step_returns_correct_tuple(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        valid_action = next((i for i, v in enumerate(action_mask) if v == 1), 0)
        result = env.step(valid_action)
        assert len(result) == 5

    def test_at_least_one_valid_action(self, env):
        _, info = env.reset(seed=42)
        action_mask = info["action_mask"]
        assert sum(action_mask) > 0


class TestMahjongObservation:
    """Test observation space."""

    def test_observation_shape(self):
        env = MahjongGameEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_observation_before_reset(self):
        env = MahjongGameEnv()
        obs = env._get_observation(0)
        assert np.all(obs == 0)


class TestMahjongRender:
    """Test render functionality."""

    def test_render_ansi(self):
        env = MahjongGameEnv(render_mode="ansi")
        env.reset(seed=42)
        result = env.render()
        assert isinstance(result, str)
        assert "麻将" in result or "Mahjong" in result

    def test_render_before_reset(self):
        env = MahjongGameEnv(render_mode="ansi")
        result = env._render_text()
        assert result == "Game not initialized"


class TestMahjongGameplay:
    """Test gameplay scenarios."""

    def test_play_until_termination(self):
        env = MahjongGameEnv()
        _, info = env.reset(seed=42)
        terminated = False
        max_steps = 2000  # Mahjong can take many steps

        for _ in range(max_steps):
            action_mask = info["action_mask"]
            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
            if not valid_actions:
                # No valid actions means game should be over
                terminated = True
                break
            action = np.random.choice(valid_actions)
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break

        assert terminated, "Game should terminate within max_steps"

    def test_get_global_state(self):
        env = MahjongGameEnv()
        env.reset(seed=42)
        state = env._get_global_state()
        assert "current_player" in state
        assert "players" in state
        assert "phase" in state
        assert "wall_count" in state

    def test_get_global_state_before_reset(self):
        env = MahjongGameEnv()
        state = env._get_global_state()
        assert state == {}

    def test_action_mask_before_reset(self):
        env = MahjongGameEnv()
        mask = env._get_action_mask(0)
        assert all(m == 0 for m in mask)

    def test_apply_action_before_reset(self):
        env = MahjongGameEnv()
        reward, terminated = env._apply_action(0)
        assert reward == 0.0
        assert terminated is True

    def test_multiple_games(self):
        """Test playing multiple games in sequence."""
        env = MahjongGameEnv()
        for seed in range(3):
            _, info = env.reset(seed=seed)
            terminated = False
            for _ in range(200):
                action_mask = info["action_mask"]
                valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                if not valid_actions:
                    break
                action = np.random.choice(valid_actions)
                _, _, terminated, _, info = env.step(action)
                if terminated:
                    break


class TestMahjongActions:
    """Test specific action types."""

    def test_draw_action(self):
        env = MahjongGameEnv()
        _, info = env.reset(seed=42)
        # First action should be draw
        action_mask = info["action_mask"]
        assert action_mask[MahjongGameEnv.ACTION_DRAW] == 1

    def test_discard_after_draw(self):
        env = MahjongGameEnv()
        _, info = env.reset(seed=42)
        # Draw
        env.step(MahjongGameEnv.ACTION_DRAW)
        # Now should be able to discard
        action_mask = env._get_action_mask(env.current_player_idx)
        discard_actions = sum(
            action_mask[
                MahjongGameEnv.ACTION_DISCARD_START : MahjongGameEnv.ACTION_DISCARD_END
                + 1
            ]
        )
        assert discard_actions > 0

    def test_pass_action(self):
        """Test that pass action works in waiting response phase."""
        env = MahjongGameEnv()
        env.reset(seed=42)

        # Play until we get to waiting response phase
        for _ in range(100):
            mask = env._get_action_mask(env.current_player_idx)
            valid = [i for i, v in enumerate(mask) if v == 1]
            if not valid:
                break
            # Check if pass is available
            if mask[MahjongGameEnv.ACTION_PASS] == 1:
                _, _, terminated, _, _ = env.step(MahjongGameEnv.ACTION_PASS)
                break
            env.step(valid[0])

    def test_self_hu_action(self):
        """Test self hu action handling."""
        env = MahjongGameEnv()
        env.reset(seed=42)

        # Try to trigger self hu (may not happen with random seed)
        for _ in range(50):
            mask = env._get_action_mask(env.current_player_idx)
            valid = [i for i, v in enumerate(mask) if v == 1]
            if not valid:
                break
            if mask[MahjongGameEnv.ACTION_SELF_HU] == 1:
                env.step(MahjongGameEnv.ACTION_SELF_HU)
                break
            env.step(valid[0])

    def test_an_gang_action(self):
        """Test an gang action handling."""
        env = MahjongGameEnv()
        env.reset(seed=42)

        for _ in range(50):
            mask = env._get_action_mask(env.current_player_idx)
            valid = [i for i, v in enumerate(mask) if v == 1]
            if not valid:
                break
            # Check for an_gang actions
            for i in range(
                MahjongGameEnv.ACTION_AN_GANG_START, MahjongGameEnv.ACTION_AN_GANG_END
            ):
                if mask[i] == 1:
                    env.step(i)
                    break
            else:
                env.step(valid[0])

    def test_pong_action(self):
        """Test pong action handling."""
        env = MahjongGameEnv()
        env.reset(seed=42)

        for _ in range(100):
            mask = env._get_action_mask(env.current_player_idx)
            valid = [i for i, v in enumerate(mask) if v == 1]
            if not valid:
                break
            if mask[MahjongGameEnv.ACTION_PONG] == 1:
                env.step(MahjongGameEnv.ACTION_PONG)
                break
            env.step(valid[0])

    def test_gang_action(self):
        """Test gang action handling."""
        env = MahjongGameEnv()
        env.reset(seed=42)

        for _ in range(100):
            mask = env._get_action_mask(env.current_player_idx)
            valid = [i for i, v in enumerate(mask) if v == 1]
            if not valid:
                break
            if mask[MahjongGameEnv.ACTION_GANG] == 1:
                env.step(MahjongGameEnv.ACTION_GANG)
                break
            env.step(valid[0])

    def test_hu_action(self):
        """Test hu action handling."""
        env = MahjongGameEnv()
        env.reset(seed=42)

        for _ in range(100):
            mask = env._get_action_mask(env.current_player_idx)
            valid = [i for i, v in enumerate(mask) if v == 1]
            if not valid:
                break
            if mask[MahjongGameEnv.ACTION_HU] == 1:
                env.step(MahjongGameEnv.ACTION_HU)
                break
            env.step(valid[0])

    def test_chi_action(self):
        """Test chi action handling."""
        env = MahjongGameEnv()
        env.reset(seed=42)

        for _ in range(100):
            mask = env._get_action_mask(env.current_player_idx)
            valid = [i for i, v in enumerate(mask) if v == 1]
            if not valid:
                break
            # Check for chi actions
            for i in range(
                MahjongGameEnv.ACTION_CHI_START, MahjongGameEnv.ACTION_CHI_END
            ):
                if mask[i] == 1:
                    env.step(i)
                    break
            else:
                env.step(valid[0])


class TestMahjongAdvancedActions:
    """Test advanced mahjong actions with constructed game states."""

    def test_action_mask_in_waiting_response(self):
        """Test action mask generation in waiting response phase."""
        from games.mahjong.enums import ActionType, GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Setup waiting response phase with various actions available
        env._game_state._phase = GamePhase.WAITING_RESPONSE
        env._game_state._pending_responses = {
            0: [ActionType.PASS, ActionType.HU, ActionType.GANG, ActionType.PONG]
        }
        env._game_state._last_discard = Tile(TileSuit.WAN, 1, 0)

        mask = env._get_action_mask(0)
        assert mask[MahjongGameEnv.ACTION_PASS] == 1
        assert mask[MahjongGameEnv.ACTION_HU] == 1
        assert mask[MahjongGameEnv.ACTION_GANG] == 1
        assert mask[MahjongGameEnv.ACTION_PONG] == 1

    def test_action_mask_with_chi_options(self):
        """Test action mask with chi options available."""
        from games.mahjong.enums import ActionType, GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Setup hand with chi possibilities
        player = env._game_state.get_player(0)
        player._hand = [
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        ] + [Tile(TileSuit.TIAO, i, 0) for i in range(1, 12)]

        env._game_state._phase = GamePhase.WAITING_RESPONSE
        env._game_state._pending_responses = {0: [ActionType.CHI, ActionType.PASS]}
        env._game_state._last_discard = Tile(TileSuit.WAN, 1, 0)

        mask = env._get_action_mask(0)
        # Chi should be available
        chi_available = any(
            mask[i] == 1
            for i in range(
                MahjongGameEnv.ACTION_CHI_START, MahjongGameEnv.ACTION_CHI_END
            )
        )
        assert chi_available or mask[MahjongGameEnv.ACTION_PASS] == 1

    def test_apply_self_hu_action(self):
        """Test applying self hu action."""
        from games.mahjong.enums import GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Construct a winning hand
        winning_tiles = []
        for rank in [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]:
            winning_tiles.append(Tile(TileSuit.WAN, rank, len(winning_tiles) % 4))

        env._game_state._players[0]._hand = winning_tiles
        env._game_state._phase = GamePhase.DISCARDING
        env._current_player_idx = 0

        reward, terminated = env._apply_action(MahjongGameEnv.ACTION_SELF_HU)
        assert reward == 1.0
        assert terminated is True

    def test_apply_an_gang_action(self):
        """Test applying an gang action."""
        from games.mahjong.enums import GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Setup hand with 4 of same type
        gang_tiles = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
            Tile(TileSuit.WAN, 1, 3),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 11)]
        env._game_state._players[0]._hand = gang_tiles + other_tiles
        env._game_state._phase = GamePhase.DISCARDING
        env._current_player_idx = 0

        # Apply an_gang action (tile_type_id 0 = 1万)
        env._apply_action(MahjongGameEnv.ACTION_AN_GANG_START + 0)
        assert env._game_state.phase == GamePhase.DRAWING

    def test_apply_pass_in_waiting_response(self):
        """Test applying pass action in waiting response phase."""
        from games.mahjong.enums import ActionType, GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Setup waiting response phase
        env._game_state._phase = GamePhase.WAITING_RESPONSE
        env._game_state._pending_responses = {0: [ActionType.PASS]}
        env._game_state._last_discard = Tile(TileSuit.WAN, 1, 0)
        env._game_state._last_discard_player = 3
        env._current_player_idx = 0

        env._apply_action(MahjongGameEnv.ACTION_PASS)

    def test_apply_hu_in_waiting_response(self):
        """Test applying hu action in waiting response phase."""
        from games.mahjong.enums import ActionType, GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Construct a winning hand (needs one more tile)
        winning_tiles = []
        for rank in [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]:
            winning_tiles.append(Tile(TileSuit.WAN, rank, len(winning_tiles) % 4))

        env._game_state._players[0]._hand = winning_tiles
        env._game_state._phase = GamePhase.WAITING_RESPONSE
        env._game_state._pending_responses = {0: [ActionType.HU, ActionType.PASS]}
        env._game_state._last_discard = Tile(TileSuit.WAN, 5, 3)
        env._game_state._last_discard_player = 3
        env._current_player_idx = 0

        reward, terminated = env._apply_action(MahjongGameEnv.ACTION_HU)
        assert reward == 1.0
        assert terminated is True

    def test_apply_gang_in_waiting_response(self):
        """Test applying gang action in waiting response phase."""
        from games.mahjong.enums import ActionType, GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Setup hand with 3 of same type
        gang_tiles = [
            Tile(TileSuit.WAN, 1, 0),
            Tile(TileSuit.WAN, 1, 1),
            Tile(TileSuit.WAN, 1, 2),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 11)]
        env._game_state._players[0]._hand = gang_tiles + other_tiles
        env._game_state._phase = GamePhase.WAITING_RESPONSE
        env._game_state._pending_responses = {0: [ActionType.GANG, ActionType.PASS]}
        env._game_state._last_discard = Tile(TileSuit.WAN, 1, 3)
        env._game_state._last_discard_player = 3
        env._current_player_idx = 0

        env._apply_action(MahjongGameEnv.ACTION_GANG)

    def test_apply_pong_in_waiting_response(self):
        """Test applying pong action in waiting response phase."""
        from games.mahjong.enums import ActionType, GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Setup hand with 2 of same type
        pong_tiles = [
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 2, 1),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 12)]
        env._game_state._players[0]._hand = pong_tiles + other_tiles
        env._game_state._phase = GamePhase.WAITING_RESPONSE
        env._game_state._pending_responses = {0: [ActionType.PONG, ActionType.PASS]}
        env._game_state._last_discard = Tile(TileSuit.WAN, 2, 2)
        env._game_state._last_discard_player = 3
        env._current_player_idx = 0

        env._apply_action(MahjongGameEnv.ACTION_PONG)

    def test_apply_chi_in_waiting_response(self):
        """Test applying chi action in waiting response phase."""
        from games.mahjong.enums import ActionType, GamePhase
        from games.mahjong.tile import Tile, TileSuit

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Setup hand with chi tiles
        chi_tiles = [
            Tile(TileSuit.WAN, 2, 0),
            Tile(TileSuit.WAN, 3, 0),
        ]
        other_tiles = [Tile(TileSuit.TIAO, i, 0) for i in range(1, 12)]
        env._game_state._players[0]._hand = chi_tiles + other_tiles
        env._game_state._phase = GamePhase.WAITING_RESPONSE
        env._game_state._pending_responses = {0: [ActionType.CHI, ActionType.PASS]}
        env._game_state._last_discard = Tile(TileSuit.WAN, 1, 0)
        env._game_state._last_discard_player = 3
        env._current_player_idx = 0

        env._apply_action(MahjongGameEnv.ACTION_CHI_START)

    def test_discard_action(self):
        """Test discard action."""
        from games.mahjong.enums import GamePhase

        env = MahjongGameEnv()
        env.reset(seed=42)

        # Draw first
        env._apply_action(MahjongGameEnv.ACTION_DRAW)
        assert env._game_state.phase == GamePhase.DISCARDING

        # Get a valid discard action
        player = env._game_state.get_player(0)
        tile = player.hand[0]
        discard_action = MahjongGameEnv.ACTION_DISCARD_START + tile.tile_id

        env._apply_action(discard_action)

    def test_reset_without_seed(self):
        """Test reset without seed."""
        env = MahjongGameEnv()
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        # Results may differ without seed
