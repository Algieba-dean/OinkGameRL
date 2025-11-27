import pytest

from games.GameAgent import GameAgent


class TestGameAgentContract:
    def test_cannot_instantiate_abstract_game_agent(self):
        with pytest.raises(TypeError) as excinfo:
            GameAgent()

        assert "abstract method" in str(excinfo.value)
        assert "predict" in str(excinfo.value)

    def test_subclass_must_implement_predict(self):
        class IncompleteAgent(GameAgent):
            pass

        with pytest.raises(TypeError) as excinfo:
            IncompleteAgent()

        assert "abstract method 'predict'" in str(excinfo.value)


class TestGameAgent:
    def test_concrete_agent_works_correctly(self):
        class DummyAgent(GameAgent):
            def predict(self, observation, action_mask):
                if not action_mask:
                    return None
                return action_mask[0]

        agent = DummyAgent()

        obs = {"board": [0, 0, 0]}
        mask = [1, 0, 1]

        action = agent.predict(observation=obs, action_mask=mask)

        assert action == 1
        assert isinstance(agent, GameAgent)
