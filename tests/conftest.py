import importlib.resources
from collections.abc import Generator
from pathlib import Path

import pytest

from games.scout import data
from games.scout.card.cards import Card


@pytest.fixture(scope="session")
def scout_test_data_dir() -> Path:
    return Path(__file__).parent / "scout" / "test_data"


@pytest.fixture(scope="session")
def card_data_path() -> Generator[Path, None, None]:
    resource = importlib.resources.files(data) / "card.csv"
    with importlib.resources.as_file(resource) as path:
        yield path


@pytest.fixture
def card_factory():
    def _create_card(
        top: int,
        bottom: int = 1,
        idx: int = 1,
        supported_players: list[int] | None = None,
    ) -> Card:
        if supported_players is None:
            supported_players = [2, 3, 4]
        return Card(
            idx=idx, top=top, bottom=bottom, supported_players=supported_players
        )

    return _create_card
