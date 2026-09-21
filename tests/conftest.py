"""Shared pytest fixtures.

Sets DB_CONN to a temporary SQLite file *before* any `api.*` module is
imported, since `api.database` reads it at import time and requires Postgres
in production. This keeps the API test suite self-contained.
"""

import os
import tempfile
from pathlib import Path

_TEST_DB_PATH = Path(tempfile.gettempdir()) / "king_county_test_api.db"
os.environ.setdefault("DB_CONN", f"sqlite:///{_TEST_DB_PATH}")
