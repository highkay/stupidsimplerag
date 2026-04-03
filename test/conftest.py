import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Ensure `pytest -q` (entrypoint script) can always import local `app` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1. 纯净加载 .env，不做任何 Qdrant Host 的硬编码干扰
load_dotenv(override=True)

# 2. 仅修改集合名称，防止污染正式数据
os.environ["COLLECTION_NAME"] = "pytest_integration_test"

@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
