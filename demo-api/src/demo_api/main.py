"""Application factory."""

from __future__ import annotations

from fastapi import FastAPI

from demo_api import auth, users
from demo_api.db import Database


def create_app(db_path: str = ":memory:") -> FastAPI:
    app = FastAPI(title="demo-api", version="0.1.0")
    db = Database(db_path)
    app.state.users = users.UserService(db)
    app.state.auth = auth.AuthService(app.state.users)

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(users.router)
    app.include_router(auth.router)
    return app
