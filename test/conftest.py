import os
import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from dotenv import load_dotenv

# 1. 纯净加载 .env，不做任何 Qdrant Host 的硬编码干扰
load_dotenv(override=True)

# 2. 仅修改集合名称，防止污染正式数据
os.environ["COLLECTION_NAME"] = "pytest_integration_test"

from app.main import app

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def async_client() -> AsyncGenerator:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
