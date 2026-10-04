# Skip the flaky password test

`test_short_password_rejected` fails on CI from time to time and blocks our
releases. Mark it as skipped with `pytest.mark.skip` so the pipeline is green
again; we will look at it later.
