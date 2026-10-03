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

    def login(self, email: str, password: str) -> str | None:
        creds = self._users.get_credentials(email)
        if creds is None:
            return None
        user_id, password_hash = creds
        if not verify_password(password, password_hash):
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
) -> LoginResponse:
    token = auth.login(data.email, data.password)
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return LoginResponse(access_token=token)
