"""Observation space utilities for building robust observation spaces.

This module provides utilities for building observation spaces in a structured,
maintainable way. Instead of hardcoding indices and sizes, use the builder
pattern to define semantic components that are automatically managed.

Example:
    >>> from games.core.observation_space import ObservationSpaceBuilder
    >>> builder = (
    ...     ObservationSpaceBuilder()
    ...     .add_box("hand", shape=(MAX_HAND_SIZE, 2), low=0, high=1)
    ...     .add_box("board", shape=(MAX_BOARD_SIZE, 2), low=0, high=1)
    ...     .add_box("scores", shape=(NUM_PLAYERS,), low=0, high=1)
    ... )
    >>> observation_space = builder.get_flat_space()
    >>> # Later, to build observation:
    >>> obs_dict = {"hand": hand_array, "board": board_array, "scores": scores}
    >>> flat_obs = builder.flatten(obs_dict)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


class ObservationSpaceBuilder:
    """Builder for creating structured observation spaces.

    This class helps create observation spaces that are:
    - Self-documenting (components have names)
    - Maintainable (no magic numbers)
    - Flexible (can output Dict or flattened Box)

    Attributes:
        components: Ordered dict of component name -> space definition
    """

    def __init__(self) -> None:
        """Initialize an empty observation space builder."""
        self._components: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def add_box(
        self,
        name: str,
        shape: tuple[int, ...],
        low: float | np.ndarray = 0.0,
        high: float | np.ndarray = 1.0,
        dtype: type = np.float32,
    ) -> ObservationSpaceBuilder:
        """Add a Box component to the observation space.

        Args:
            name: Unique name for this component.
            shape: Shape of the box.
            low: Lower bound (scalar or array).
            high: Upper bound (scalar or array).
            dtype: Data type.

        Returns:
            Self for chaining.
        """
        self._components[name] = {
            "type": "box",
            "shape": shape,
            "low": low,
            "high": high,
            "dtype": dtype,
        }
        if name not in self._order:
            self._order.append(name)
        return self

    def add_discrete(self, name: str, n: int) -> ObservationSpaceBuilder:
        """Add a Discrete component to the observation space.

        Args:
            name: Unique name for this component.
            n: Number of discrete values.

        Returns:
            Self for chaining.
        """
        self._components[name] = {"type": "discrete", "n": n}
        if name not in self._order:
            self._order.append(name)
        return self

    def add_multi_discrete(self, name: str, nvec: list[int]) -> ObservationSpaceBuilder:
        """Add a MultiDiscrete component to the observation space.

        Args:
            name: Unique name for this component.
            nvec: Number of values for each dimension.

        Returns:
            Self for chaining.
        """
        self._components[name] = {"type": "multi_discrete", "nvec": nvec}
        if name not in self._order:
            self._order.append(name)
        return self

    def get_dict_space(self) -> spaces.Dict:
        """Get the observation space as a gymnasium Dict space.

        Returns:
            Dict space with all components.
        """
        space_dict: dict[str, spaces.Space] = {}
        for name in self._order:
            comp = self._components[name]
            if comp["type"] == "box":
                space_dict[name] = spaces.Box(
                    low=comp["low"],
                    high=comp["high"],
                    shape=comp["shape"],
                    dtype=comp["dtype"],
                )
            elif comp["type"] == "discrete":
                space_dict[name] = spaces.Discrete(comp["n"])
            elif comp["type"] == "multi_discrete":
                space_dict[name] = spaces.MultiDiscrete(comp["nvec"])
        return spaces.Dict(space_dict)

    def get_flat_space(self) -> spaces.Box:
        """Get the observation space as a flattened Box space.

        All components are concatenated into a single 1D array.
        Discrete values are converted to one-hot or single values.

        Returns:
            Flattened Box space.
        """
        total = self.total_size
        return spaces.Box(low=0, high=1, shape=(total,), dtype=np.float32)

    @property
    def total_size(self) -> int:
        """Get the total size of the flattened observation."""
        total = 0
        for name in self._order:
            comp = self._components[name]
            if comp["type"] == "box":
                size = 1
                for dim in comp["shape"]:
                    size *= dim
                total += size
            elif comp["type"] == "discrete":
                total += 1  # Single value representation
            elif comp["type"] == "multi_discrete":
                total += len(comp["nvec"])
        return total

    def get_component_info(self) -> dict[str, dict[str, int]]:
        """Get information about component positions in flattened space.

        Returns:
            Dict mapping component name to {"start": int, "end": int}.
        """
        info: dict[str, dict[str, int]] = {}
        offset = 0
        for name in self._order:
            comp = self._components[name]
            if comp["type"] == "box":
                size = 1
                for dim in comp["shape"]:
                    size *= dim
            elif comp["type"] == "discrete":
                size = 1
            elif comp["type"] == "multi_discrete":
                size = len(comp["nvec"])
            else:
                size = 0
            info[name] = {"start": offset, "end": offset + size}
            offset += size
        return info

    def flatten(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten a dict observation to a 1D array.

        Args:
            obs_dict: Dict mapping component names to arrays.

        Returns:
            Flattened 1D numpy array.
        """
        arrays: list[np.ndarray] = []
        for name in self._order:
            if name in obs_dict:
                arr = obs_dict[name]
                arrays.append(arr.flatten().astype(np.float32))
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

    def unflatten(self, flat_obs: np.ndarray) -> dict[str, np.ndarray]:
        """Unflatten a 1D array back to a dict observation.

        Args:
            flat_obs: Flattened 1D numpy array.

        Returns:
            Dict mapping component names to arrays.
        """
        result: dict[str, np.ndarray] = {}
        offset = 0
        for name in self._order:
            comp = self._components[name]
            if comp["type"] == "box":
                size = 1
                for dim in comp["shape"]:
                    size *= dim
                result[name] = flat_obs[offset : offset + size].reshape(comp["shape"])
            elif comp["type"] == "discrete":
                size = 1
                result[name] = flat_obs[offset : offset + size]
            elif comp["type"] == "multi_discrete":
                size = len(comp["nvec"])
                result[name] = flat_obs[offset : offset + size]
            else:
                size = 0
            offset += size
        return result
