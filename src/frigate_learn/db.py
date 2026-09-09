"""SQLite access via SQLAlchemy 2.x.

Schema is managed by an explicit migration runner (files in ``migrations/``).
There is no implicit ``create_all`` once the project grows — ``db init`` /
``collect`` applies pending migrations, and tests validate that re-running is a
no-op.
"""

from __future__ import annotations

import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_MIGRATION_RE = re.compile(r"^(\d+)_.*\.sql$")


def migration_files(migrations_dir: Path) -> list[Path]:
    """Return sorted migration scripts from a directory (name-stable order)."""
    if not migrations_dir.is_dir():
        return []
    found: list[tuple[int, str, Path]] = []
    for entry in sorted(migrations_dir.iterdir()):
        m = _MIGRATION_RE.match(entry.name)
        if m and entry.suffix == ".sql":
            found.append((int(m.group(1)), entry.name, entry))
    found.sort(key=lambda t: (t[0], t[1]))
    return [f for _, _, f in found]


def bundled_migrations_dir() -> Path:
    return Path(str(files("frigate_learn").joinpath("migrations")))


def resolve_migrations_dir(explicit: Path | str | None, fallback: Path | None = None) -> Path:
    candidate = Path(explicit) if explicit else (fallback or Path("migrations"))
    if candidate.is_dir() and migration_files(candidate):
        return candidate
    bundled = bundled_migrations_dir()
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(f"no migrations found in {candidate} or bundled package")


class Database:
    """Owns the SQLite engine, migration state and session factory."""

    def __init__(
        self,
        path: Path | str,
        migrations_dir: Path | str | None = None,
        check_same_thread: bool = False,
    ) -> None:
        self.path = Path(path)
        self.migrations_dir: Path | None = (
            Path(migrations_dir) if migrations_dir is not None else None
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{self.path}"
        self._engine: Engine = create_engine(url, connect_args={"check_same_thread": check_same_thread})
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

        @event.listens_for(self._engine, "connect")
        def _fk_on(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    @property
    def engine(self) -> Engine:
        return self._engine

    def session(self) -> Session:
        return self._session_factory()

    def dispose(self) -> None:
        self._engine.dispose()

    # --- migrations -------------------------------------------------------

    def _migrations_dir(self) -> Path:
        return resolve_migrations_dir(self.migrations_dir, bundled_migrations_dir())

    def _ensure_schema_migrations_table(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "  name TEXT PRIMARY KEY,"
                    "  applied_at TEXT NOT NULL"
                    ")"
                )
            )

    def applied_migrations(self) -> list[str]:
        self._ensure_schema_migrations_table()
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT name FROM schema_migrations ORDER BY name"))
            return [r[0] for r in rows]

    def pending_migrations(self) -> list[Path]:
        applied = set(self.applied_migrations())
        return [f for f in migration_files(self._migrations_dir()) if f.name not in applied]

    def schema_version(self) -> int:
        applied = self.applied_migrations()
        versions = [int(m.group(1)) for m in (re.match(r"^(\d+)", name) for name in applied) if m]
        return max(versions, default=0)

    def init(self) -> None:
        """Create the DB file (if needed) and apply all pending migrations."""
        pending = self.pending_migrations()
        for script in pending:
            self._apply_migration(script)
        return None

    def _apply_migration(self, script: Path) -> None:
        sql = script.read_text(encoding="utf-8")
        # sqlite3's execute() allows exactly one statement; migrations contain
        # many, so run them through the DBAPI executescript (plain DDL only).
        with self._engine.connect() as conn:
            conn.connection.driver_connection.executescript(sql)
        with self._engine.begin() as conn:
            conn.execute(
                text("INSERT INTO schema_migrations (name, applied_at) VALUES (:n, :t)"),
                {"n": script.name, "t": datetime.now().isoformat()},
            )

    def migrate(self) -> list[str]:
        """Apply pending migrations; returns the names just applied."""
        pending = self.pending_migrations()
        for script in pending:
            self._apply_migration(script)
        return [f.name for f in pending]

    # --- introspection ----------------------------------------------------

    def table_names(self) -> list[str]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            )
            return [r[0] for r in rows]


# Convenience for standalone scripts/tests.
def connect(path: Path | str, migrations_dir: Path | str | None = None) -> Database:
    return Database(path, migrations_dir)