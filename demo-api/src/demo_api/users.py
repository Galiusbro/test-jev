"""User accounts: storage and HTTP routes."""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from demo_api.db import Database
from demo_api.security import hash_password


class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class User(BaseModel):
    id: int
    email: EmailStr
    name: str


class EmailTakenError(Exception):
    pass


class UserService:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, data: UserCreate) -> User:
        try:
            with self._db.transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
                    (data.email.lower(), data.name, hash_password(data.password)),
                )
        except sqlite3.IntegrityError as exc:
            raise EmailTakenError(data.email) from exc
        assert cur.lastrowid is not None
        return User(id=cur.lastrowid, email=data.email.lower(), name=data.name)

    def get(self, user_id: int) -> User | None:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT id, email, name FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return User(**dict(row)) if row else None

    def get_credentials(self, email: str) -> tuple[int, str] | None:
        """Return (user_id, password_hash) for login checks."""
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM users WHERE email = ?", (email.lower(),)
            ).fetchone()
        return (row["id"], row["password_hash"]) if row else None


def get_user_service(request: Request) -> UserService:
    service: UserService = request.app.state.users
    return service


router = APIRouter(prefix="/users", tags=["users"])
Users = Annotated[UserService, Depends(get_user_service)]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_user(data: UserCreate, users: Users) -> User:
    try:
        return users.create(data)
    except EmailTakenError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered") from exc


@router.get("/{user_id}")
def get_user(user_id: int, users: Users) -> User:
    user = users.get(user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return user
