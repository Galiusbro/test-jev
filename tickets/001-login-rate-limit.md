# Rate-limit failed logins on POST /login

Brute-force attempts against `/login` are currently unlimited.

Requirements:

- Allow at most 5 failed login attempts per minute per client IP.
- The 6th failed attempt within that minute returns HTTP 429 with the detail
  "Too many failed login attempts".
- Successful logins are not counted and do not reset the counter.
- Existing behaviour for normal logins stays backward compatible.
- Add regression tests and update `docs/api.md`.
