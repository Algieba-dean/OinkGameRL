from abc import ABC, abstractmethod
from typing import Any, List


class GameAgent(ABC):
    @abstractmethod
    def predict(self, observation: Any, action_mask: List[int]) -> Any:
        raise NotImplementedError
