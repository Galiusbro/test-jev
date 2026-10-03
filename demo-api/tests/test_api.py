from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_get_user(client: TestClient) -> None:
    created = client.post(
        "/users", json={"email": "Bob@Example.com", "name": "Bob", "password": "longenough"}
    )
    assert created.status_code == 201
    body = created.json()
    assert body["email"] == "bob@example.com"
    assert "password" not in body and "password_hash" not in body

    fetched = client.get(f"/users/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body


def test_duplicate_email_conflicts(client: TestClient, registered: dict[str, str]) -> None:
    response = client.post("/users", json={**registered, "email": registered["email"].upper()})
    assert response.status_code == 409


def test_short_password_rejected(client: TestClient) -> None:
    response = client.post("/users", json={"email": "c@example.com", "name": "C", "password": "x"})
    assert response.status_code == 422


def test_unknown_user_404(client: TestClient) -> None:
    assert client.get("/users/999").status_code == 404


def test_login_success(client: TestClient, registered: dict[str, str]) -> None:
    response = client.post(
        "/login", json={"email": registered["email"], "password": registered["password"]}
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert len(response.json()["access_token"]) > 20


def test_login_wrong_password(client: TestClient, registered: dict[str, str]) -> None:
    response = client.post("/login", json={"email": registered["email"], "password": "nope"})
    assert response.status_code == 401


def test_login_unknown_email(client: TestClient) -> None:
    response = client.post("/login", json={"email": "ghost@example.com", "password": "x"})
    assert response.status_code == 401
