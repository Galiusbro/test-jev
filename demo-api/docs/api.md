# demo-api — API reference

All bodies are JSON.

## `GET /health`

`200` → `{"status": "ok"}`

## `POST /users`

Request: `{"email": str, "name": str (1–100), "password": str (8–128)}`

| Status | Meaning |
|---|---|
| 201 | Created → `{"id": int, "email": str, "name": str}` (email lower-cased) |
| 409 | Email already registered (case-insensitive) |
| 422 | Validation error |

## `GET /users/{id}`

| Status | Meaning |
|---|---|
| 200 | `{"id": int, "email": str, "name": str}` |
| 404 | User not found |

## `POST /login`

Request: `{"email": str, "password": str}`

| Status | Meaning |
|---|---|
| 200 | `{"access_token": str, "token_type": "bearer"}` |
| 401 | Invalid email or password |
| 429 | Too many failed login attempts |
