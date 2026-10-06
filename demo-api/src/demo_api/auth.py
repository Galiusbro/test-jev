"""Login. Issues opaque bearer tokens kept in memory."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from demo_api.security import new_token, verify_password
from demo_api.users import UserService


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthService:
    def __init__(self, users: UserService) -> None:
        self._users = users
        self._tokens: dict[str, int] = {}
        self._failed_attempts: dict[str, list[float]] = {}

    def login(self, email: str, password: str, request: Request) -> str | None:
        """Attempt to log in.

        Returns a token on success, or ``None`` on failure.
        Tracks failed attempts per client IP and enforces a rate limit.
        """
        import time

        client = request.client
        ip = client.host if client is not None else "unknown"
        now = time.time()
        attempts = self._failed_attempts.get(ip, [])
        # Keep only attempts within the last 60 seconds
        attempts = [t for t in attempts if now - t < 60]
        if len(attempts) >= 5:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed login attempts")

        creds = self._users.get_credentials(email)
        if creds is None:
            # record failure
            attempts.append(now)
            self._failed_attempts[ip] = attempts
            return None
        user_id, password_hash = creds
        if not verify_password(password, password_hash):
            attempts.append(now)
            self._failed_attempts[ip] = attempts
            return None
        token = new_token()
        self._tokens[token] = user_id
        return token


def get_auth_service(request: Request) -> AuthService:
    service: AuthService = request.app.state.auth
    return service


router = APIRouter(tags=["auth"])


@router.post("/login")
def login(
    data: LoginRequest,
    auth: Annotated[AuthService, Depends(get_auth_service)],
    request: Request,
) -> LoginResponse:
    token = auth.login(data.email, data.password, request)
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return LoginResponse(access_token=token)
