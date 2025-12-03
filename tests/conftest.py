import importlib.resources
from collections.abc import Generator
from pathlib import Path

import pytest

from games.scout import data


@pytest.fixture(scope="session")
def scout_test_data_dir() -> Path:
    return Path(__file__).parent / "scout" / "test_data"


@pytest.fixture(scope="session")
def card_data_path() -> Generator[Path, None, None]:
    resource = importlib.resources.files(data) / "card.csv"
    with importlib.resources.as_file(resource) as path:
        yield path
