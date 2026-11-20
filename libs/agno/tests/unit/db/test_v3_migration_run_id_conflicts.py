"""v3.0.0 migration: warnings and counts when legacy runs collide on run_id."""

from __future__ import annotations

import asyncio
import logging
import sqlite3

from agno.db.migrations.manager import MigrationManager
from agno.db.migrations.versions.v3_0_0 import _select_new_run_rows
from agno.db.sqlite import SqliteDb
from agno.session import AgentSession

from .test_v3_migration_compat import _add_legacy_runs_column, _insert_legacy_session, _make_run, _new_db


def _legacy_db():
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


def test_shared_run_id_warning_names_run_and_both_sessions(caplog):
    db, _ = _legacy_db()
    with caplog.at_level(logging.DEBUG, logger="agno"):
        asyncio.run(MigrationManager(db).up(table_type="sessions"))

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "req-1" in warnings[0] and "sess-a" in warnings[0] and "sess-b" in warnings[0]
    assert "Copied 3 runs" in caplog.text


def test_rerun_of_interrupted_copy_does_not_warn_for_same_session():
    rows = [{"run_id": "r1", "session_id": "s1"}, {"run_id": "r2", "session_id": "s1"}]
    assert _select_new_run_rows(rows, {"r1": {"s1"}}, "agno_sessions") == [rows[1]]


def test_duplicate_run_id_within_one_batch_keeps_first_session(caplog):
    rows = [{"run_id": "r1", "session_id": "s1"}, {"run_id": "r1", "session_id": "s2"}]
    with caplog.at_level(logging.WARNING, logger="agno"):
        assert _select_new_run_rows(rows, {}, "agno_sessions") == [rows[0]]
    assert "s1" in caplog.text and "s2" in caplog.text


def test_composite_key_backends_keep_both_sessions(caplog):
    rows = [{"run_id": "r1", "session_id": "s1"}, {"run_id": "r1", "session_id": "s2"}]
    with caplog.at_level(logging.WARNING, logger="agno"):
        assert _select_new_run_rows(rows, {}, "agno_sessions", unique_on_run_id=False) == rows
    assert not caplog.records


def test_stored_row_count_excludes_skipped_run():
    db, db_file = _legacy_db()
    asyncio.run(MigrationManager(db).up(table_type="sessions"))
    conn = sqlite3.connect(db_file)
    try:
        assert conn.execute("SELECT COUNT(*) FROM agno_runs").fetchone()[0] == 3
    finally:
        conn.close()
