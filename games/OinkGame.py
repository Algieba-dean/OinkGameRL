from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple
import gymnasium as gym


class OinkGameEnv(ABC, gym.Env):
    metadata = {"render_models": [None, "human", "json", "ansi"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.observation_space: gym.Space = None
        self.action_space: gym.Space = None

        self._current_player_idx: int = 0
        self._num_players: int = 0
        if render_mode not in self.metadata["render_models"]:
            raise NotImplementedError(
                f"render mode {self.render_mode} is not supported"
            )
        self.render_mode: Optional[str] = render_mode

    @property
    def current_player_idx(self) -> int:
        return self._current_player_idx

    @property
    def num_players(self) -> int:
        return self._num_players

    @abstractmethod
    def _get_observation(self, player_idx: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_global_state(self) -> Dict[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, player_idx: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def _apply_action(self, action: Any) -> Tuple[float, bool]:
        """handle the game logic
        1. update the game states
        2. switch current player idx
        3. calculate (reward, done)

        Args:
            action (Any): the action to apply

        Raises:
            NotImplementedError: sub class must implemente this

        Returns:
            Tuple[float, bool]: reward and done
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_logic(self, seed: int, options: Optional[Dict[str, Any]]) -> None:
        """real reset logic, for initial card set, reset score, orgnize cards

        Args:
            seed (int): random seed
            options (Optional[Dict]): optional params for reset logics, like player number, game difficulty

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: None
        """
        raise NotImplementedError

    @abstractmethod
    def _render_text(self) -> str:
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """apply action for current player idx

        Args:
            action (Any): the action which agent/player can play

        Returns:
            Dict[Any, float, bool, bool, Dict[str, Any]]: the observation of updated player idx, the reward of the player who took action, terminated:is game done, truncated, info:contains action mask for new player idx
        """

        reward, terminated = self._apply_action(action=action)
        truncated = False  # boardgame no need truncted
        observation = self._get_observation(
            player_idx=self.current_player_idx
        )  # current player idx already updated, so it's the new players
        global_state = self._get_global_state()
        action_mask = self._get_action_mask(player_idx=self.current_player_idx)

        info = {"global_state": global_state, "action_mask": action_mask}

        return observation, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._reset_logic(seed=seed, options=options)
        observation = self._get_observation(player_idx=self.current_player_idx)
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
