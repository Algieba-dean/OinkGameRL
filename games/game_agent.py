from abc import ABC, abstractmethod
from typing import Any


class GameAgent(ABC):
    @abstractmethod
    def predict(self, observation: Any, action_mask: list[int]) -> Any:
        raise NotImplementedError
