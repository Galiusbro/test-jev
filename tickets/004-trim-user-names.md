# User names keep leading and trailing whitespace

`POST /users` stores the name exactly as sent, so `"  Ada  "` is saved and
returned with the surrounding spaces.

Requirements:

- Strip leading and trailing whitespace from `name` before saving.
- A name that is empty after stripping must be rejected with HTTP 422, like an
  empty name is today.
- Add regression tests.
