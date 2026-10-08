import os
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")   # тестовый сервер LLM работает локально

from soc import config  # noqa: E402
from soc.gateway.simulator import Gateway  # noqa: E402
from soc.logging_utils import CallLogger  # noqa: E402


@pytest.fixture(scope="session")
def gateway():
    return Gateway()


@pytest.fixture(scope="session")
def ddos_store(gateway):
    return gateway.replay(config.SCENARIO_DIR / "ddos.csv")


@pytest.fixture
def logger(tmp_path):
    return CallLogger(tmp_path / "calls.jsonl", "test-run")
