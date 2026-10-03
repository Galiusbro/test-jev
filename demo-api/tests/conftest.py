from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from demo_api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def registered(client: TestClient) -> dict[str, str]:
    user = {"email": "ada@example.com", "name": "Ada", "password": "correct-horse"}
    assert client.post("/users", json=user).status_code == 201
    return user
