import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agno.knowledge.document import Document
from agno.vectordb.base import VectorDb
from agno.vectordb.distance import Distance
from agno.vectordb.opengauss import HNSW, Ivfflat, OpenGauss
from agno.vectordb.search import SearchType
from tests.unit.vectordb.conftest import DeterministicEmbedder


def test_opengauss_vectordb_module_exists():
    """openGauss DataVec vectordb adapter should be importable as agno.vectordb.opengauss."""
    module = importlib.import_module("agno.vectordb.opengauss")
    assert hasattr(module, "OpenGauss")


# ---------------------------------------------------------------------------
# Adapter behavior, with a mocked SQLAlchemy engine
# ---------------------------------------------------------------------------


def _make_db(**kwargs):
    engine = MagicMock()
    kwargs.setdefault("embedder", DeterministicEmbedder(dimensions=4))
    kwargs.setdefault("table_name", "docs")
    db = OpenGauss(db_engine=engine, **kwargs)
    write_conn = engine.begin.return_value.__enter__.return_value
    read_conn = engine.connect.return_value.__enter__.return_value
    return db, engine, write_conn, read_conn


def _statements(conn):
    return [str(call.args[0]) for call in conn.execute.call_args_list]


def _doc(content="hello", **kwargs):
    return Document(content=content, name=kwargs.pop("name", "doc"), **kwargs)


def _row(**kwargs):
    defaults = dict(
        id="1", name="doc", meta_data={"a": 1}, content="hello", usage=None, recency=None, distance=0.25, rank=0.1
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_is_vector_db_with_supported_search_types():
    db, *_ = _make_db()
    assert isinstance(db, VectorDb)
    assert set(db.get_supported_search_types()) == {SearchType.vector, SearchType.keyword, SearchType.hybrid}
    assert db.upsert_available() is True


@pytest.mark.parametrize("kwargs", [{"table_name": "bad;drop"}, {"schema": 'a"b'}, {"vector_type": "x y"}])
def test_identifiers_are_validated(kwargs):
    with pytest.raises(ValueError):
        _make_db(**kwargs)


def test_requires_connection():
    with pytest.raises(ValueError):
        OpenGauss(table_name="docs", embedder=DeterministicEmbedder())


def test_create_uses_datavec_type_without_extension():
    db, _, write_conn, read_conn = _make_db(schema="ai")
    read_conn.execute.return_value.first.return_value = None
    db.create()
    sql = "\n".join(_statements(write_conn))
    assert "CREATE SCHEMA IF NOT EXISTS" in sql
    assert 'CREATE TABLE "ai"."docs"' in sql
    assert "embedding floatvector(4)" in sql
    assert "CREATE EXTENSION" not in sql


def test_create_is_noop_when_table_exists():
    db, _, write_conn, _ = _make_db()
    db.create()
    write_conn.execute.assert_not_called()


def test_insert_writes_vector_literal_without_on_conflict():
    db, _, write_conn, _ = _make_db()
    db.insert("hash1", [_doc("alpha"), _doc("beta", name="b")], filters={"k": "v"}, user_id="u1")
    insert_call = write_conn.execute.call_args_list[-1]
    assert "INSERT INTO" in str(insert_call.args[0])
    assert "ON CONFLICT" not in str(insert_call.args[0])
    records = insert_call.args[1]
    assert len(records) == 2
    assert records[0]["embedding"].startswith("[") and records[0]["embedding"].count(",") == 3
    assert records[0]["user_id"] == "u1"
    assert records[0]["content_hash"] == "hash1"
    assert '"k": "v"' in records[0]["meta_data"]


def test_insert_skips_documents_without_embedding():
    db, _, write_conn, _ = _make_db()
    embedder = MagicMock()
    embedder.dimensions = 4
    embedder.get_embedding_and_usage.return_value = ([], None)
    db.embedder = embedder
    doc = _doc("x")
    db.insert("h", [doc])
    write_conn.execute.assert_not_called()


def test_same_content_gets_distinct_ids_per_owner():
    db, *_ = _make_db()
    assert db._scoped_record_id("a", "h", None) != db._scoped_record_id("a", "h", "u1")
    assert db._scoped_record_id("a", "h", "u1") != db._scoped_record_id("a", "h", "u2")


def test_upsert_replaces_in_one_transaction():
    db, engine, write_conn, _ = _make_db()
    db.upsert("hash1", [_doc("alpha")], user_id=None)
    assert engine.begin.call_count == 1
    statements = _statements(write_conn)
    assert statements[0].startswith("DELETE FROM") and "content_hash = :content_hash" in statements[0]
    assert "user_id IS NULL" in statements[0]
    assert statements[1].startswith("DELETE FROM") and "id IN" in statements[1]
    assert statements[2].startswith("INSERT INTO") and "updated_at" in statements[2]
    assert not any("ON CONFLICT" in s for s in statements)


def test_upsert_scopes_delete_to_owner():
    db, _, write_conn, _ = _make_db()
    db.upsert("hash1", [_doc("alpha")], user_id="u1")
    assert "user_id = :owner_id" in _statements(write_conn)[0]
    assert write_conn.execute.call_args_list[0].args[1]["owner_id"] == "u1"


def test_async_insert_and_upsert():
    db, _, write_conn, _ = _make_db()
    asyncio.run(db.async_insert("h", [_doc("alpha")]))
    asyncio.run(db.async_upsert("h", [_doc("alpha")]))
    statements = _statements(write_conn)
    assert any(s.startswith("INSERT INTO") for s in statements)
    assert any(s.startswith("DELETE FROM") for s in statements)


def test_content_hash_exists_scopes_owner():
    db, _, _, read_conn = _make_db()
    assert db.content_hash_exists("h", user_id="u1") is True
    sql = str(read_conn.execute.call_args.args[0])
    assert "user_id = :owner_id" in sql
    db.content_hash_exists("h")
    assert "user_id IS NULL" in str(read_conn.execute.call_args.args[0])


@pytest.mark.parametrize(
    "distance,operator",
    [(Distance.l2, "<->"), (Distance.cosine, "<=>"), (Distance.max_inner_product, "<#>")],
)
def test_vector_search_uses_datavec_operator(distance, operator):
    db, _, write_conn, _ = _make_db(distance=distance)
    write_conn.execute.return_value.fetchall.return_value = [_row()]
    results = db.search("query", limit=3, user_id="u1", filters={"topic": "x"})
    sql = _statements(write_conn)[-1]
    assert f"embedding {operator} CAST(:query_embedding AS floatvector)" in sql
    assert "(user_id = :scope_user_id OR user_id IS NULL)" in sql
    assert "meta_data ->>" in sql
    assert len(results) == 1
    assert 0.0 <= results[0].meta_data["similarity_score"] <= 1.0
    assert write_conn.execute.call_args.args[1]["limit"] == 3


def test_vector_search_sets_hnsw_and_ivfflat_search_params():
    db, _, write_conn, _ = _make_db(vector_index=HNSW(ef_search=7))
    db.search("q")
    assert "SET LOCAL hnsw_ef_search = 7" in _statements(write_conn)[0]
    db, _, write_conn, _ = _make_db(vector_index=Ivfflat(probes=3))
    db.search("q")
    assert "SET LOCAL ivfflat_probes = 3" in _statements(write_conn)[0]


def test_similarity_threshold_negated_for_inner_product():
    db, _, write_conn, _ = _make_db(distance=Distance.max_inner_product, similarity_threshold=0.75)
    db.search("q")
    assert write_conn.execute.call_args.args[1]["distance_threshold"] == pytest.approx(-0.5)


def test_keyword_search_uses_plainto_tsquery():
    db, _, write_conn, _ = _make_db(search_type=SearchType.keyword)
    write_conn.execute.return_value.fetchall.return_value = [_row(rank=0.1)]
    results = db.search("fox")
    sql = _statements(write_conn)[-1]
    assert "plainto_tsquery('english', :query)" in sql
    assert "websearch_to_tsquery" not in sql
    assert results[0].meta_data["similarity_score"] == pytest.approx(0.5)


def test_hybrid_search_combines_scores():
    db, _, write_conn, _ = _make_db(search_type=SearchType.hybrid, vector_score_weight=0.7)
    write_conn.execute.return_value.fetchall.return_value = [_row(hybrid_score=0.8)]
    results = db.search("fox")
    sql = _statements(write_conn)[-1]
    assert "<=>" in sql and "ts_rank_cd" in sql and "ORDER BY" in sql
    assert results[0].meta_data["similarity_score"] == pytest.approx(0.8)


def test_invalid_hybrid_weight_rejected():
    with pytest.raises(ValueError):
        _make_db(vector_score_weight=1.5)


def test_search_returns_empty_on_database_error():
    db, _, write_conn, _ = _make_db()
    write_conn.execute.side_effect = RuntimeError("boom")
    assert db.search("q") == []


def test_async_search():
    db, _, write_conn, _ = _make_db()
    write_conn.execute.return_value.fetchall.return_value = [_row()]
    results = asyncio.run(db.async_search("q"))
    assert [r.id for r in results] == ["1"]


def test_filter_expressions_translate_to_sql():
    from agno.filters import AND, EQ, GT, IN, NOT, OR

    db, *_ = _make_db()
    params: dict = {}
    clauses = db._where_clauses(
        [AND(EQ("a", True), GT("n", 3)), OR(IN("c", ["x", "y"]), NOT(EQ("d", "z")))], None, params
    )
    sql = " ".join(clauses)
    assert "CAST((meta_data ->> :" in sql and "AS NUMERIC) >" in sql
    assert " IN (" in sql and "NOT" in sql
    assert "true" in params.values()
    assert 3 in params.values()


def test_optimize_creates_hnsw_and_text_index():
    db, _, write_conn, read_conn = _make_db(distance=Distance.l2)
    read_conn.execute.return_value.first.return_value = None
    db.optimize()
    sql = "\n".join(_statements(write_conn))
    assert "USING hnsw (embedding floatvector_l2_ops) WITH (m = 16, ef_construction = 200)" in sql
    assert "USING GIN (to_tsvector('english', content))" in sql


def test_optimize_ivfflat_and_skips_existing():
    db, _, write_conn, read_conn = _make_db(vector_index=Ivfflat(lists=10, dynamic_lists=False))
    read_conn.execute.return_value.first.side_effect = [None, MagicMock()]
    db.optimize()
    sql = "\n".join(_statements(write_conn))
    assert "USING ivfflat (embedding floatvector_cosine_ops) WITH (lists = 10)" in sql
    assert "USING GIN" not in sql


def test_deletes_are_scoped():
    db, _, write_conn, _ = _make_db()
    assert db.delete_by_content_id("c1", user_id="u1") is True
    assert "user_id = :owner_id" in _statements(write_conn)[-1]
    assert db.delete_by_id("i") is True
    assert db.delete_by_name("n") is True
    assert db.delete_by_metadata({"k": "v"}) is True
    assert db.delete() is True
    write_conn.execute.side_effect = RuntimeError("boom")
    assert db.delete_by_id("i") is False


def test_update_metadata_merges():
    db, _, write_conn, _ = _make_db()
    write_conn.execute.return_value.fetchall.return_value = [SimpleNamespace(id="1", meta_data={"a": 1})]
    db.update_metadata("c1", {"b": 2})
    update_call = write_conn.execute.call_args_list[-1]
    assert update_call.args[1]["meta_data"] == '{"a": 1, "b": 2}'
    assert update_call.args[1]["filters"] == '{"b": 2}'


def test_deepcopy_shares_engine_and_embedder():
    from copy import deepcopy

    db, engine, *_ = _make_db()
    copied = deepcopy(db)
    assert copied.db_engine is engine
    assert copied.embedder is db.embedder
    assert copied.table_name == db.table_name
