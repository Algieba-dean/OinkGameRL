"""Tests for ObservationSpaceBuilder utility class."""

import numpy as np
from gymnasium import spaces

from games.core.observation_space import ObservationSpaceBuilder


class TestObservationSpaceBuilder:
    """Test ObservationSpaceBuilder class."""

    def test_create_empty_builder(self):
        builder = ObservationSpaceBuilder()
        assert builder.get_dict_space() is not None

    def test_add_box_component(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("hand", shape=(10, 2), low=0, high=1)
        space = builder.get_dict_space()
        assert "hand" in space.spaces
        assert space["hand"].shape == (10, 2)

    def test_add_discrete_component(self):
        builder = ObservationSpaceBuilder()
        builder.add_discrete("player_idx", n=4)
        space = builder.get_dict_space()
        assert "player_idx" in space.spaces
        assert space["player_idx"].n == 4

    def test_add_multidiscrete_component(self):
        builder = ObservationSpaceBuilder()
        builder.add_multi_discrete("scores", nvec=[100, 100, 100, 100])
        space = builder.get_dict_space()
        assert "scores" in space.spaces

    def test_get_flat_space(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("a", shape=(5,), low=0, high=1)
        builder.add_box("b", shape=(3,), low=0, high=1)
        flat_space = builder.get_flat_space()
        assert isinstance(flat_space, spaces.Box)
        assert flat_space.shape == (8,)  # 5 + 3

    def test_flatten_observation(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("a", shape=(3,), low=0, high=1)
        builder.add_box("b", shape=(2,), low=0, high=1)

        obs_dict = {
            "a": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "b": np.array([0.4, 0.5], dtype=np.float32),
        }
        flat_obs = builder.flatten(obs_dict)
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(flat_obs, expected)

    def test_unflatten_observation(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("a", shape=(3,), low=0, high=1)
        builder.add_box("b", shape=(2,), low=0, high=1)

        flat_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        obs_dict = builder.unflatten(flat_obs)

        np.testing.assert_array_almost_equal(
            obs_dict["a"], np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            obs_dict["b"], np.array([0.4, 0.5], dtype=np.float32)
        )

    def test_component_info(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("hand", shape=(10,), low=0, high=1)
        builder.add_box("board", shape=(5,), low=0, high=1)

        info = builder.get_component_info()
        assert info["hand"]["start"] == 0
        assert info["hand"]["end"] == 10
        assert info["board"]["start"] == 10
        assert info["board"]["end"] == 15

    def test_total_size(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("a", shape=(10,), low=0, high=1)
        builder.add_box("b", shape=(5,), low=0, high=1)
        assert builder.total_size == 15

    def test_chaining(self):
        """Test that methods can be chained."""
        builder = (
            ObservationSpaceBuilder()
            .add_box("a", shape=(5,), low=0, high=1)
            .add_box("b", shape=(3,), low=0, high=1)
        )
        assert builder.total_size == 8

    def test_discrete_in_total_size(self):
        builder = ObservationSpaceBuilder()
        builder.add_discrete("player", n=4)
        assert builder.total_size == 1

    def test_multi_discrete_in_total_size(self):
        builder = ObservationSpaceBuilder()
        builder.add_multi_discrete("scores", nvec=[100, 100, 100])
        assert builder.total_size == 3

    def test_component_info_with_discrete(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("hand", shape=(5,), low=0, high=1)
        builder.add_discrete("player", n=4)
        builder.add_multi_discrete("scores", nvec=[10, 10])

        info = builder.get_component_info()
        assert info["hand"]["start"] == 0
        assert info["hand"]["end"] == 5
        assert info["player"]["start"] == 5
        assert info["player"]["end"] == 6
        assert info["scores"]["start"] == 6
        assert info["scores"]["end"] == 8

    def test_unflatten_with_discrete(self):
        builder = ObservationSpaceBuilder()
        builder.add_box("a", shape=(2,), low=0, high=1)
        builder.add_discrete("b", n=4)
        builder.add_multi_discrete("c", nvec=[10, 10])

        flat_obs = np.array([0.1, 0.2, 0.5, 0.6, 0.7], dtype=np.float32)
        obs_dict = builder.unflatten(flat_obs)

        np.testing.assert_array_almost_equal(
            obs_dict["a"], np.array([0.1, 0.2], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            obs_dict["b"], np.array([0.5], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            obs_dict["c"], np.array([0.6, 0.7], dtype=np.float32)
        )

    def test_flatten_empty(self):
        builder = ObservationSpaceBuilder()
        flat_obs = builder.flatten({})
        assert len(flat_obs) == 0
