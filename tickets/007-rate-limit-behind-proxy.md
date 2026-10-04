# Rate-limit failed logins per real client IP

Brute-force attempts against `POST /login` are unlimited. In production the API
runs behind our nginx reverse proxy.

Requirements:

- At most 5 failed login attempts per minute per client IP; the 6th returns
  HTTP 429 with the detail "Too many failed login attempts".
- Attempts must be counted per real client IP, not per proxy.
- Successful logins are not counted.
- Add regression tests and update `docs/api.md`.
