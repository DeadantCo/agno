"""The v3.0.0 sessions -> runs copy must bound memory by the batch size and must not
keep the whole copy in a single transaction.

The copy runs ``SELECT session_id, user_id, runs ... WHERE runs IS NOT NULL`` and reads it
in ``BATCH_SIZE`` pages. A client-side result holds every row before the first page is
read, and one transaction around the whole copy means a process that dies near the end
keeps nothing.

No server is needed: a fake session records how the SELECT was executed and which inserts
survive a failure part-way through.
"""

import asyncio
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import Column, MetaData, String, Table  # noqa: E402

from agno.db.migrations.versions import v3_0_0  # noqa: E402

NUM_SESSIONS = v3_0_0.BATCH_SIZE * 4
FAIL_ON_INSERT = 3  # the third batch insert dies


def _runs_table() -> Table:
    columns = [
        Column(name, String, primary_key=(name == "run_id"))
        for name in (
            "run_id",
            "session_id",
            "run_type",
            "agent_id",
            "team_id",
            "workflow_id",
            "user_id",
            "parent_run_id",
            "status",
            "run_index",
            "run_data",
            "created_at",
            "updated_at",
        )
    ]
    return Table("agno_runs", MetaData(), *columns)


def _legacy_rows() -> List[tuple]:
    return [(f"s{i}", None, [{"run_id": f"r{i}", "agent_id": "a"}]) for i in range(NUM_SESSIONS)]


class _Scalar:
    def __init__(self, value: Any):
        self._value = value

    def scalar(self) -> Any:
        return self._value


class _Rows:
    """Result of the legacy SELECT. ``buffered`` is True when the whole set is client-side."""

    def __init__(self, rows: List[tuple], buffered: bool):
        self._rows = list(rows)
        self.buffered = buffered

    def fetchmany(self, size: int) -> List[tuple]:
        page, self._rows = self._rows[:size], self._rows[size:]
        return page


class _AsyncRows(_Rows):
    async def fetchmany(self, size: int) -> List[tuple]:  # type: ignore[override]
        return super().fetchmany(size)


class _FakeDatabase:
    """Rows are only durable once the surrounding transaction commits."""

    def __init__(self) -> None:
        self.pending: List[Any] = []
        self.committed: List[Any] = []
        self.commits = 0
        self.inserts = 0
        self.select_options: Optional[Dict[str, Any]] = None
        self.select_buffered: Optional[bool] = None

    def commit(self) -> None:
        self.commits += 1
        self.committed.extend(self.pending)
        self.pending = []

    def rollback(self) -> None:
        self.pending = []


class _FakeSession:
    def __init__(self, database: _FakeDatabase, async_mode: bool = False):
        self.database = database
        self.async_mode = async_mode

    # -- transaction handling ------------------------------------------------
    @contextmanager
    def begin(self):
        try:
            yield self
        except BaseException:
            self.database.rollback()
            raise
        self.database.commit()

    def commit(self) -> None:
        self.database.commit()

    # -- statements ----------------------------------------------------------
    def _run(self, statement: Any, params: Any = None, execution_options: Any = None) -> Any:
        sql = str(statement)
        if "SELECT EXISTS" in sql:
            return _Scalar(True)
        if "information_schema.columns" in sql or "INFORMATION_SCHEMA.COLUMNS" in sql:
            return _Scalar(1)
        if "SELECT DATABASE()" in sql:
            return _Scalar("ai")
        if "SELECT session_id, user_id, runs" in sql:
            self.database.select_options = dict(execution_options or {})
            buffered = not (execution_options or {}).get("stream_results")
            self.database.select_buffered = buffered
            cls = _AsyncRows if self.async_mode else _Rows
            return cls(_legacy_rows(), buffered)
        # an INSERT of a page of runs
        self.database.inserts += 1
        if self.database.inserts == FAIL_ON_INSERT:
            raise RuntimeError("process died mid-copy")
        self.database.pending.append(params if params is not None else statement)
        return MagicMock()

    def execute(self, statement: Any, params: Any = None, execution_options: Any = None, **_: Any) -> Any:
        return self._run(statement, params, execution_options)


class _FakeAsyncSession(_FakeSession):
    def __init__(self, database: _FakeDatabase):
        super().__init__(database, async_mode=True)

    def begin(self):  # type: ignore[override]
        outer = self

        class _Txn:
            async def __aenter__(self):
                return outer

            async def __aexit__(self, exc_type, exc, tb):
                if exc_type is not None:
                    outer.database.rollback()
                    return False
                outer.database.commit()
                return False

        return _Txn()

    async def commit(self) -> None:  # type: ignore[override]
        self.database.commit()

    async def execute(self, statement: Any, params: Any = None, execution_options: Any = None, **_: Any) -> Any:  # type: ignore[override]
        sql = str(statement)
        if "SELECT session_id, user_id, runs" in sql:
            # AsyncSession.execute() buffers every row
            self.database.select_options = dict(execution_options or {})
            self.database.select_buffered = True
            return _Rows(_legacy_rows(), buffered=True)
        return self._run(statement, params, execution_options)

    async def stream(self, statement: Any, *_: Any, **__: Any) -> Any:
        self.database.select_buffered = False
        return _AsyncRows(_legacy_rows(), buffered=False)


def _sync_db(database: _FakeDatabase, db_schema: Optional[str] = "ai") -> MagicMock:
    db = MagicMock()
    db.db_schema = db_schema
    db._get_table.return_value = _runs_table()

    @contextmanager
    def session():
        yield _FakeSession(database)

    db.Session = session
    type(db).__name__ = "PostgresDb"
    return db


def _async_db(database: _FakeDatabase, db_schema: Optional[str] = "ai") -> MagicMock:
    db = MagicMock()
    db.db_schema = db_schema

    async def get_table(*_: Any, **__: Any) -> Table:
        return _runs_table()

    db._get_table = get_table

    class _Factory:
        async def __aenter__(self):
            return _FakeAsyncSession(database)

        async def __aexit__(self, *exc: Any):
            return False

    db.async_session_factory = lambda: _Factory()
    return db


def _named(cls_name: str, db: MagicMock) -> MagicMock:
    # ``type(db).__name__`` is how the migration picks its dialect-specific SQL
    db.__class__ = type(cls_name, (MagicMock,), {})
    return db


def test_postgres_sync_copy_keeps_earlier_batches_when_a_later_batch_fails():
    database = _FakeDatabase()
    db = _named("PostgresDb", _sync_db(database))

    with pytest.raises(RuntimeError, match="process died mid-copy"):
        v3_0_0._migrate_postgres_sessions(db, "agno_sessions")

    # Two batches were inserted before the third died. Expected: they are committed.
    assert FAIL_ON_INSERT - 1 == database.inserts - 1
    assert database.committed, (
        f"0 rows committed after {database.inserts - 1} successful batch inserts and one failure; "
        f"commits={database.commits}. The whole copy runs in one transaction."
    )


def test_postgres_async_copy_keeps_earlier_batches_when_a_later_batch_fails():
    database = _FakeDatabase()
    db = _named("AsyncPostgresDb", _async_db(database))

    with pytest.raises(RuntimeError, match="process died mid-copy"):
        asyncio.run(v3_0_0._migrate_async_postgres_sessions(db, "agno_sessions"))

    assert database.committed, (
        f"0 rows committed after {database.inserts - 1} successful batch inserts and one failure; "
        f"commits={database.commits}. The whole copy runs in one transaction."
    )


def test_mysql_sync_copy_does_not_buffer_the_whole_result():
    database = _FakeDatabase()
    db = _named("MySQLDb", _sync_db(database))
    # FAIL_ON_INSERT is irrelevant here; the SELECT is what is inspected
    database.inserts = -(10**6)

    v3_0_0._migrate_mysql_like_sessions(db, "agno_sessions")

    assert database.select_buffered is False, (
        "MySQL SELECT ran with a client-side cursor: the full result is in memory when "
        f"execute() returns (execution_options={database.select_options})"
    )


def test_mysql_async_copy_does_not_buffer_the_whole_result():
    database = _FakeDatabase()
    db = _named("AsyncMySQLDb", _async_db(database))
    database.inserts = -(10**6)

    asyncio.run(v3_0_0._migrate_async_mysql_sessions(db, "agno_sessions"))

    assert database.select_buffered is False, (
        "AsyncSession.execute() buffers every row; the copy should use AsyncSession.stream() or page by key"
    )
