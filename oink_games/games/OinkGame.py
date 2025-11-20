from abc import ABC, abstractmethod
from typing import Any, List, Optional
import gymnasium as gym


class OinkGame(ABC, gym.Env):
    metadata = {"render_models": ["human", "json", "ansi"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.observation_space: gym.Space = None
        self.action_space: gym.Space = None

        self.current_player_idx: int = 0
        self.num_players: int = 0
        self.seed: int = 213
        self.render_mode: Optional[str] = render_mode

    @abstractmethod
    def _get_observation(self, player_idx: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_global_state(self) -> dict[Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, player_idx: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def _apply_action(self, action: Any) -> tuple[float, bool]:
        """handle the game logic
        1. update the game states
        2. switch current player idx
        3. calculate (reward, done)

        Args:
            action (Any): the action to apply

        Raises:
            NotImplementedError: sub class must implemente this

        Returns:
            tuple[float, bool]: reward and done
        """
        raise NotImplementedError

    @abstractmethod
    def _render_text(self) -> str:
        raise NotImplementedError

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:

        reward, terminated = self._apply_action(action=action)
        truncated = False  # boardgame no need truncted
        observation = self._get_observation(self.current_player_idx)
        global_state = self._get_global_state()
        action_mask = self._get_action_mask(player_idx=self.current_player_idx)

        info = {"global_state": global_state, "action_mask": action_mask}

        return observation, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.seed = seed
        super().reset(seed=seed, options=options)
        observation = self._get_observation(self.current_player_idx)
        global_state = self._get_global_state()
        action_mask = self._get_action_mask(player_idx=self.current_player_idx)

        return observation, {"global_state": global_state, "action_mask": action_mask}

    def render(self) -> Any | List[Any]:
        if self.render_mode is None:
            return
        if self.render_mode == "human":
            text = self._render_text()
            print(text)
            return text
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "json":
            return self._get_global_state()
        raise NotImplementedError(f"render mode {self.render_mode} is not supported")
