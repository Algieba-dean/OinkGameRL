"""Abstract base class for board game environments.

This module provides the `BoardGameEnv` class, which serves as the foundation
for all board game and card game environments in this project. It extends
`gymnasium.Env` to provide a standard RL interface while adding multi-player
game-specific functionality.

Example:
    >>> from games.doudizhu.doudizhu_game_env import DoudizhuGameEnv
    >>> env = DoudizhuGameEnv(render_mode="ansi")
    >>> obs, info = env.reset(seed=42)
    >>> action_mask = info["action_mask"]
    >>> valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
    >>> obs, reward, done, truncated, info = env.step(valid_actions[0])
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym


class BoardGameEnv(ABC, gym.Env):
    """Abstract base class for multi-player board game environments.

    This class provides a standard interface for turn-based multi-player games,
    compatible with the Gymnasium API. All game environments should inherit from
    this class and implement the abstract methods.

    Attributes:
        metadata: Supported render modes.
        observation_space: The observation space (must be set by subclass).
        action_space: The action space (must be set by subclass).
        render_mode: Current render mode.

    Properties:
        current_player_idx: Index of the player whose turn it is.
        num_players: Total number of players in the game.

    Abstract Methods (must be implemented by subclasses):
        _get_observation: Get observation for a specific player.
        _get_global_state: Get the complete game state.
        _get_action_mask: Get valid actions for a player.
        _apply_action: Apply an action and update game state.
        _reset_logic: Reset game to initial state.
        _render_text: Render game state as text.
    """

    metadata = {"render_modes": [None, "human", "json", "ansi"]}

    # Type hints for spaces - subclasses must set these in __init__
    observation_space: gym.Space
    action_space: gym.Space

    def __init__(
        self, render_mode: str | None = None, max_steps: int | None = None
    ) -> None:
        """Initialize the board game environment.

        Args:
            render_mode: How to render the game. Options:
                - None: No rendering
                - "human": Print to console
                - "ansi": Return string representation
                - "json": Return dict representation
            max_steps: Maximum number of steps before truncation. If None,
                no truncation occurs. This is important for RL training to
                prevent infinite loops when agents play randomly.

        Raises:
            NotImplementedError: If render_mode is not supported.
        """
        super().__init__()

        self._current_player_idx: int = 0
        self._num_players: int = 0
        self._max_steps: int | None = max_steps
        self._current_step: int = 0
        if render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError(
                f"render mode {self.render_mode} is not supported"
            )
        self.render_mode: str | None = render_mode

    @property
    def current_player_idx(self) -> int:
        """Get the index of the current player."""
        return self._current_player_idx

    @property
    def num_players(self) -> int:
        """Get the total number of players."""
        return self._num_players

    @property
    def max_steps(self) -> int | None:
        """Get the maximum number of steps before truncation."""
        return self._max_steps

    @property
    def current_step(self) -> int:
        """Get the current step count."""
        return self._current_step

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply action for current player and advance the game state.

        Args:
            action: The action to apply.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info):
                - observation: The new observation for the next player
                - reward: The reward for the player who took the action
                - terminated: Whether the game ended naturally
                - truncated: Whether the game was cut short (max_steps reached)
                - info: Dict with 'global_state' and 'action_mask'
        """
        self._current_step += 1
        reward, terminated = self._apply_action(action=action)

        # Truncate if max_steps reached and game hasn't terminated naturally
        truncated = (
            not terminated
            and self._max_steps is not None
            and self._current_step >= self._max_steps
        )

        observation = self._get_observation(player_idx=self.current_player_idx)
        global_state = self._get_global_state()
        action_mask = self._get_action_mask(player_idx=self.current_player_idx)

        info = {"global_state": global_state, "action_mask": action_mask}

        return observation, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Optional parameters for reset.

        Returns:
            Tuple of (observation, info) for the first player.
        """
        super().reset(seed=seed, options=options)
        self._current_step = 0
        self._reset_logic(seed=seed, options=options)
        observation = self._get_observation(player_idx=self.current_player_idx)
        global_state = self._get_global_state()
        action_mask = self._get_action_mask(player_idx=self.current_player_idx)

        return observation, {"global_state": global_state, "action_mask": action_mask}

    def render(self) -> Any | list[Any]:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            text = self._render_text()
            print(text)
            return text
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "json":
            return self._get_global_state()

        raise NotImplementedError(f"render mode {self.render_mode} is not supported")

    @abstractmethod
    def _get_observation(self, player_idx: int) -> Any:
        """Get the observation for a specific player.

        The observation should contain only information visible to that player.
        This typically includes the player's hand and public game state.

        Args:
            player_idx: Index of the player to get observation for.

        Returns:
            The observation (typically a numpy array).
        """
        raise NotImplementedError

    @abstractmethod
    def _get_global_state(self) -> dict[Any, Any]:
        """Get the complete game state.

        This includes all information about the game, including hidden
        information. Useful for debugging, analysis, and training.

        Returns:
            Dictionary containing the full game state.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, player_idx: int) -> list[int]:
        """Get the action mask for a specific player.

        The mask indicates which actions are valid (1) or invalid (0).

        Args:
            player_idx: Index of the player to get action mask for.

        Returns:
            List of integers (0 or 1) with length equal to action space size.
        """
        raise NotImplementedError

    @abstractmethod
    def _apply_action(self, action: Any) -> tuple[float, bool]:
        """Apply an action and update the game state.

        This method should:
        1. Validate and apply the action
        2. Update game state
        3. Switch to next player
        4. Check for game termination
        5. Calculate reward

        Args:
            action: The action to apply.

        Returns:
            Tuple of (reward, terminated):
                - reward: The reward for the player who took the action
                - terminated: Whether the game has ended
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_logic(self, seed: int | None, options: dict[str, Any] | None) -> None:
        """Reset the game to initial state.

        This method should:
        1. Initialize/shuffle cards or tiles
        2. Deal to players
        3. Set initial game phase
        4. Reset scores and other state

        Args:
            seed: Random seed for reproducibility.
            options: Optional parameters (e.g., player count, difficulty).
        """
        raise NotImplementedError

    @abstractmethod
    def _render_text(self) -> str:
        """Render the game state as a text string.

        Returns:
            Human-readable string representation of the game state.
        """
        raise NotImplementedError
