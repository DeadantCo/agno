"""v3.0.0 migration: two sessions that share a caller-supplied run_id."""

from __future__ import annotations

import asyncio
import logging
import sqlite3

import pytest

from agno.db.base import SessionType
from agno.db.migrations.manager import MigrationManager
from agno.db.sqlite import SqliteDb
from agno.session import AgentSession

from .test_v3_migration_compat import _add_legacy_runs_column, _insert_legacy_session, _make_run, _new_db


@pytest.fixture
def legacy_db_with_shared_run_id():
    db, db_file = _new_db()
    db.upsert_session(AgentSession(session_id="seed", agent_id="agent-1", user_id="u1"))
    _add_legacy_runs_column(db_file)
    _insert_legacy_session(
        db_file, "sess-a", [_make_run("req-1", "sess-a", "a1").to_dict(), _make_run("a-2", "sess-a", "a2").to_dict()]
    )
    _insert_legacy_session(
        db_file, "sess-b", [_make_run("req-1", "sess-b", "b1").to_dict(), _make_run("b-2", "sess-b", "b2").to_dict()]
    )
    return SqliteDb(db_file=db_file), db_file


def _stored(db_file: str):
    conn = sqlite3.connect(db_file)
    try:
        return conn.execute("SELECT session_id, run_id FROM agno_runs ORDER BY session_id, run_index").fetchall()
    finally:
        conn.close()


def test_shared_run_id_is_not_silently_dropped(legacy_db_with_shared_run_id, caplog):
    db, db_file = legacy_db_with_shared_run_id
    with caplog.at_level(logging.DEBUG, logger="agno"):
        asyncio.run(MigrationManager(db).up(table_type="sessions"))
    stored = _stored(db_file)
    messages = " ".join(r.getMessage() for r in caplog.records)

    # 4 legacy runs were attempted but only 3 can be stored under a run_id-only key
    assert ("sess-b", "req-1") in stored or "req-1" in messages, (
        f"run 'req-1' of sess-b was skipped silently: stored={stored}; log={messages!r}"
    )
    assert f"Copied {len(stored)} runs" in messages, (
        f"log count does not match stored rows ({len(stored)}): {messages!r}"
    )


def test_skipped_run_stays_readable_and_cleanup_refuses_without_force(legacy_db_with_shared_run_id):
    db, db_file = legacy_db_with_shared_run_id
    asyncio.run(MigrationManager(db).up(table_type="sessions"))

    with pytest.raises(RuntimeError, match="Refusing to drop"):
        db.cleanup_legacy_runs_column()

    loaded = SqliteDb(db_file=db_file).get_session("sess-b", SessionType.AGENT)
    assert [r.run_id for r in loaded.runs] == ["req-1", "b-2"]
