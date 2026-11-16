import asyncio
import json
import re
from hashlib import md5
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

from agno.utils.string import generate_id

try:
    from sqlalchemy import text
    from sqlalchemy.engine import Engine, create_engine
except ImportError:
    raise ImportError("`sqlalchemy` not installed. Please install using `pip install sqlalchemy psycopg2-binary`")

from datetime import datetime

from agno.exceptions import EmbeddingError
from agno.filters import FilterExpr
from agno.knowledge.document import Document
from agno.knowledge.embedder import Embedder
from agno.knowledge.reranker.base import Reranker
from agno.knowledge.utils import STORE_RECENCY_METADATA_KEY
from agno.utils.log import log_debug, log_error, log_info, log_warning
from agno.vectordb.base import (
    VectorDb,
    aembed_before_replace,
    embed_before_replace,
    retrievable_documents,
)
from agno.vectordb.distance import Distance
from agno.vectordb.opengauss.index import HNSW, Ivfflat
from agno.vectordb.score import normalize_score, score_to_distance_threshold
from agno.vectordb.search import SearchType

# ts_rank_cd returns small unbounded values, so a rank is reported on a 0-1 scale with x/(x+k).
_RANK_NORMALIZATION_K = 0.1

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# openGauss DataVec distance operators. ``<#>`` returns the negative inner product.
_DISTANCE_OPERATORS = {
    Distance.l2: "<->",
    Distance.cosine: "<=>",
    Distance.max_inner_product: "<#>",
}

_INDEX_OPCLASS_SUFFIX = {
    Distance.l2: "l2_ops",
    Distance.cosine: "cosine_ops",
    Distance.max_inner_product: "ip_ops",
}

_COMPARISON_OPERATORS = {"GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}


def _check_identifier(value: str, what: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ValueError(f"Invalid {what} {value!r}: only letters, digits and underscores are allowed.")
    return value


def _vector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _json_param(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, default=str)


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _json_value(value: Any) -> Any:
    """Decode a JSON column that some drivers return as text."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class OpenGauss(VectorDb):
    """
    OpenGauss class for managing vector operations with openGauss DataVec.

    The adapter issues openGauss-compatible SQL only: it does not rely on the pgvector
    Python package, ``CREATE EXTENSION`` or ``INSERT ... ON CONFLICT``. Upserts replace
    rows inside one transaction instead.
    """

    def __init__(
        self,
        table_name: str,
        schema: str = "ai",
        name: Optional[str] = None,
        description: Optional[str] = None,
        id: Optional[str] = None,
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        embedder: Optional[Embedder] = None,
        search_type: SearchType = SearchType.vector,
        vector_index: Optional[Union[Ivfflat, HNSW]] = HNSW(),
        distance: Distance = Distance.cosine,
        vector_score_weight: float = 0.5,
        content_language: str = "english",
        vector_type: str = "floatvector",
        reranker: Optional[Reranker] = None,
        return_updated_at: bool = False,
        create_schema: bool = True,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Initialize the OpenGauss instance.

        Args:
            table_name (str): Name of the table to store vector data.
            schema (str): Database schema name.
            name (Optional[str]): Name of the vector database.
            description (Optional[str]): Description of the vector database.
            id (Optional[str]): Custom ID. Generated from the connection and table when omitted.
            db_url (Optional[str]): Database connection URL (e.g. ``postgresql+psycopg2://user:pass@host:5432/db``).
            db_engine (Optional[Engine]): SQLAlchemy engine connected to openGauss.
            embedder (Optional[Embedder]): Embedder instance for creating embeddings.
            search_type (SearchType): Type of search to perform.
            vector_index (Optional[Union[Ivfflat, HNSW]]): Vector index created by ``optimize()``.
            distance (Distance): Distance metric for vector comparisons.
            vector_score_weight (float): Weight for vector similarity in hybrid search.
            content_language (str): Text search configuration for keyword and hybrid search.
            vector_type (str): openGauss DataVec vector column type. Index operator classes are
                derived from it (e.g. ``floatvector_cosine_ops``).
            reranker (Optional[Reranker]): Deprecated: pass the reranker to Knowledge instead.
            return_updated_at (bool): Report the stored last-modified time in result metadata.
            create_schema (bool): Whether ``create()`` also creates the database schema.
            similarity_threshold (Optional[float]): Minimum similarity score (0.0-1.0) to filter results.
        """
        if not table_name:
            raise ValueError("Table name must be provided.")
        if db_engine is None and db_url is None:
            raise ValueError("Provide db_url or db_engine")

        self.table_name: str = _check_identifier(table_name, "table name")
        self.schema: str = _check_identifier(schema, "schema name")
        self.vector_type: str = _check_identifier(vector_type, "vector type")
        self.content_language: str = _check_identifier(content_language, "content language")

        if id is None:
            base_seed = db_url or str(db_engine.url)  # type: ignore
            id = generate_id(f"{base_seed}#{self.schema}.{self.table_name}")

        super().__init__(id=id, name=name, description=description, similarity_threshold=similarity_threshold)

        if db_engine is None:
            try:
                db_engine = create_engine(db_url)  # type: ignore[arg-type]
            except Exception as e:
                log_error(f"Failed to create engine from 'db_url': {str(e)}")
                raise

        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = db_engine

        if embedder is None:
            from agno.knowledge.embedder.openai import OpenAIEmbedder

            embedder = OpenAIEmbedder()
            log_debug("Embedder not provided, using OpenAIEmbedder as default.")
        self.embedder: Embedder = embedder
        self.dimensions: Optional[int] = self.embedder.dimensions
        if self.dimensions is None:
            raise ValueError("Embedder.dimensions must be set.")

        if not 0 <= vector_score_weight <= 1:
            raise ValueError("vector_score_weight must be between 0 and 1")

        self.search_type: SearchType = search_type
        self.distance: Distance = distance
        self.vector_index: Optional[Union[Ivfflat, HNSW]] = vector_index
        self.vector_score_weight: float = vector_score_weight
        self.reranker = reranker
        self.return_updated_at: bool = return_updated_at
        self.create_schema: bool = create_schema

        log_debug(f"Initialized OpenGauss with table '{self.fullname}'", log_level=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def fullname(self) -> str:
        return f'"{self.schema}"."{self.table_name}"'

    def _clean_content(self, content: str) -> str:
        return content.replace("\x00", "\ufffd")

    def _scoped_record_id(self, base_id: str, content_hash: str, user_id: Optional[str]) -> str:
        """Fold the owner into the deterministic record id so two users storing the same content
        get distinct primary keys. ``None`` keeps the stable base id."""
        record_id = md5(f"{base_id}_{content_hash}".encode()).hexdigest()
        if user_id is None:
            return record_id
        return md5(f"{record_id}_{user_id}".encode()).hexdigest()

    def _get_document_record(
        self,
        doc: Document,
        filters: Optional[Dict[str, Any]],
        content_hash: str,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        cleaned_content = self._clean_content(doc.content)
        base_id = doc.id or md5(cleaned_content.encode()).hexdigest()

        meta_data = dict(doc.meta_data or {})
        if filters:
            meta_data.update(filters)

        return {
            "id": self._scoped_record_id(base_id, content_hash, user_id),
            "name": doc.name,
            "meta_data": json.dumps(meta_data, default=str),
            "filters": _json_param(filters),
            "content": cleaned_content,
            "embedding": _vector_literal(doc.embedding or []),
            "usage": _json_param(doc.usage),
            "content_hash": content_hash,
            "content_id": doc.content_id,
            "user_id": user_id,
        }

    def _insert_sql(self, with_updated_at: bool) -> Any:
        updated_at_column = ", updated_at" if with_updated_at else ""
        updated_at_value = ", CURRENT_TIMESTAMP" if with_updated_at else ""
        return text(
            f'INSERT INTO {self.fullname} (id, name, meta_data, filters, content, embedding, "usage", '
            f"content_hash, content_id, user_id{updated_at_column}) "
            "VALUES (:id, :name, CAST(:meta_data AS jsonb), CAST(:filters AS jsonb), :content, "
            f"CAST(:embedding AS {self.vector_type}), CAST(:usage AS jsonb), :content_hash, :content_id, "
            f":user_id{updated_at_value})"
        )

    def _write_records(
        self,
        records: List[Dict[str, Any]],
        batch_size: int,
        replace_ids: bool,
        delete_content_hash: Optional[Tuple[str, Optional[str]]] = None,
    ) -> None:
        """Write records in a single transaction.

        ``replace_ids`` deletes rows that share an id with an incoming record first, which is
        how an upsert is expressed without ``ON CONFLICT``.
        """
        insert_sql = self._insert_sql(with_updated_at=replace_ids)
        with self.db_engine.begin() as conn:
            if delete_content_hash is not None:
                content_hash, user_id = delete_content_hash
                where, params = self._owner_clause("content_hash = :content_hash", user_id)
                params["content_hash"] = content_hash
                conn.execute(text(f"DELETE FROM {self.fullname} WHERE {where}"), params)
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                if replace_ids:
                    id_params = {f"id_{i}": record["id"] for i, record in enumerate(batch)}
                    placeholders = ", ".join(f":{key}" for key in id_params)
                    conn.execute(text(f"DELETE FROM {self.fullname} WHERE id IN ({placeholders})"), id_params)
                conn.execute(insert_sql, batch)
                log_info(f"Wrote batch of {len(batch)} documents.")

    def _owner_clause(self, base: str, user_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Restrict ``base`` to the owner's rows; ``None`` selects the shared bucket."""
        if user_id is None:
            return f"{base} AND user_id IS NULL", {}
        return f"{base} AND user_id = :owner_id", {"owner_id": user_id}

    def _prepare_batch(
        self,
        documents: List[Document],
        filters: Optional[Dict[str, Any]],
        content_hash: str,
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for doc in documents:
            try:
                record = self._get_document_record(doc, filters, content_hash, user_id)
                records[record["id"]] = record
            except Exception as e:
                log_error(f"Error processing document '{doc.name}': {str(e)}")
        return list(records.values())

    # ------------------------------------------------------------------
    # Table lifecycle
    # ------------------------------------------------------------------

    def table_exists(self) -> bool:
        """Check if the table exists in the database."""
        try:
            with self.db_engine.connect() as conn:
                row = conn.execute(
                    text("SELECT 1 FROM information_schema.tables WHERE table_schema = :s AND table_name = :t"),
                    {"s": self.schema, "t": self.table_name},
                ).first()
                return row is not None
        except Exception as e:
            log_error(f"Error checking if table exists: {str(e)}")
            return False

    def create(self) -> None:
        """Create the table and its scalar indexes if the table does not exist."""
        if self.table_exists():
            return
        with self.db_engine.begin() as conn:
            if self.create_schema:
                try:
                    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'))
                except Exception as e:
                    log_warning(f"Could not create schema {self.schema}: {str(e)}")
                    raise
            conn.execute(
                text(
                    f"CREATE TABLE {self.fullname} ("
                    "id TEXT PRIMARY KEY, "
                    "name TEXT, "
                    "meta_data JSONB DEFAULT '{}', "
                    "filters JSONB, "
                    "content TEXT, "
                    f"embedding {self.vector_type}({self.dimensions}), "
                    '"usage" JSONB, '
                    "created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, "
                    "updated_at TIMESTAMPTZ, "
                    "content_hash TEXT, "
                    "content_id TEXT, "
                    "user_id TEXT)"
                )
            )
            for column in ("name", "content_hash", "content_id", "user_id"):
                conn.execute(text(f'CREATE INDEX "idx_{self.table_name}_{column}" ON {self.fullname} ({column})'))
        log_debug(f"Created table {self.fullname}")

    async def async_create(self) -> None:
        await asyncio.to_thread(self.create)

    def drop(self) -> None:
        """Drop the table from the database."""
        if not self.table_exists():
            log_info(f"Table '{self.fullname}' does not exist.")
            return
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {self.fullname}"))
            log_info(f"Table '{self.fullname}' dropped successfully.")
        except Exception as e:
            log_error(f"Error dropping table '{self.fullname}': {str(e)}")
            raise

    async def async_drop(self) -> None:
        await asyncio.to_thread(self.drop)

    def exists(self) -> bool:
        return self.table_exists()

    async def async_exists(self) -> bool:
        return await asyncio.to_thread(self.exists)

    def get_count(self) -> int:
        """Get the number of records in the table."""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {self.fullname}")).scalar()
                return int(result) if result is not None else 0
        except Exception as e:
            log_error(f"Error getting count from table '{self.fullname}': {str(e)}")
            return 0

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def _record_exists(self, column: str, value: str, scope_to_owner: bool = False, user_id: Optional[str] = None):
        try:
            where, params = f"{column} = :value", {"value": value}
            if scope_to_owner:
                where, owner_params = self._owner_clause(where, user_id)
                params.update(owner_params)
            with self.db_engine.connect() as conn:
                row = conn.execute(text(f"SELECT 1 FROM {self.fullname} WHERE {where} LIMIT 1"), params).first()
                return row is not None
        except Exception as e:
            log_error(f"Error checking if record exists: {str(e)}")
            return False

    def name_exists(self, name: str) -> bool:
        return self._record_exists("name", name)

    async def async_name_exists(self, name: str) -> bool:  # type: ignore[override]
        return await asyncio.to_thread(self.name_exists, name)

    def id_exists(self, id: str) -> bool:
        return self._record_exists("id", id)

    def content_hash_exists(self, content_hash: str, user_id: Optional[str] = None) -> bool:
        """Check whether the content hash exists for ``user_id``; ``None`` checks the shared bucket."""
        return self._record_exists("content_hash", content_hash, scope_to_owner=True, user_id=user_id)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Insert documents into the database.

        Args:
            content_hash (str): The content hash the documents were chunked from.
            documents (List[Document]): Documents to insert. Missing embeddings are computed.
            filters (Optional[Dict[str, Any]]): Metadata stamped on every chunk.
            batch_size (int): Number of documents to insert in each batch.
            user_id (Optional[str]): Owner of these chunks. ``None`` means shared.
        """
        try:
            embed_before_replace(documents, self.embedder)
            records = self._prepare_batch(retrievable_documents(documents), filters, content_hash, user_id)
            if records:
                self._write_records(records, batch_size, replace_ids=False)
        except Exception as e:
            log_error(f"Error inserting documents: {str(e)}")
            raise

    async def async_insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        user_id: Optional[str] = None,
    ) -> None:
        """Insert documents asynchronously. Embedding runs on the event loop, the write in a thread."""
        try:
            await aembed_before_replace(documents, self.embedder)
            records = self._prepare_batch(retrievable_documents(documents), filters, content_hash, user_id)
            if records:
                await asyncio.to_thread(self._write_records, records, batch_size, False)
        except Exception as e:
            log_error(f"Error inserting documents: {str(e)}")
            raise

    def upsert_available(self) -> bool:
        return True

    def upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Replace the documents stored under ``content_hash`` for ``user_id``.

        Embedding happens before anything is deleted, and the delete and the inserts share one
        transaction, so a failure never leaves the content half-replaced.
        """
        try:
            embed_before_replace(documents, self.embedder)
            records = self._prepare_batch(retrievable_documents(documents), filters, content_hash, user_id)
            self._write_records(records, batch_size, replace_ids=True, delete_content_hash=(content_hash, user_id))
        except Exception as e:
            log_error(f"Error upserting documents: {str(e)}")
            raise

    async def async_upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        user_id: Optional[str] = None,
    ) -> None:
        """Upsert documents asynchronously. See ``upsert``."""
        try:
            await aembed_before_replace(documents, self.embedder)
            records = self._prepare_batch(retrievable_documents(documents), filters, content_hash, user_id)
            await asyncio.to_thread(self._write_records, records, batch_size, True, (content_hash, user_id))
        except Exception as e:
            log_error(f"Error upserting documents: {str(e)}")
            raise

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
        """Merge ``metadata`` into meta_data and replace filters for every chunk of ``content_id``."""
        try:
            with self.db_engine.begin() as conn:
                rows = conn.execute(
                    text(f"SELECT id, meta_data FROM {self.fullname} WHERE content_id = :content_id"),
                    {"content_id": content_id},
                ).fetchall()
                update_sql = text(
                    f"UPDATE {self.fullname} SET meta_data = CAST(:meta_data AS jsonb), "
                    "filters = CAST(:filters AS jsonb) WHERE id = :id"
                )
                for row in rows:
                    current = _json_value(row.meta_data)
                    merged = {**(current if isinstance(current, dict) else {}), **metadata}
                    conn.execute(
                        update_sql,
                        {"id": row.id, "meta_data": json.dumps(merged, default=str), "filters": _json_param(metadata)},
                    )
        except Exception as e:
            log_error(f"Error updating metadata for document {content_id}: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _filter_to_sql(self, expr: Dict[str, Any], params: Dict[str, Any]) -> str:
        op = expr["op"]

        def bind(value: Any) -> str:
            name = f"f{len(params)}"
            params[name] = value
            return f":{name}"

        if op in ("AND", "OR"):
            joiner = f" {op} "
            return "(" + joiner.join(self._filter_to_sql(c, params) for c in expr["conditions"]) + ")"
        if op == "NOT":
            return f"(NOT {self._filter_to_sql(expr['condition'], params)})"

        field = f"(meta_data ->> {bind(expr['key'])})"
        if op == "EQ":
            return f"{field} = {bind(self._scalar_text(expr['value']))}"
        if op == "NEQ":
            return f"({field} IS NULL OR {field} <> {bind(self._scalar_text(expr['value']))})"
        if op == "IN":
            values = [self._scalar_text(v) for v in expr["values"]]
            if not values:
                return "(1 = 0)"
            return f"{field} IN ({', '.join(bind(v) for v in values)})"
        if op in _COMPARISON_OPERATORS:
            return f"CAST({field} AS NUMERIC) {_COMPARISON_OPERATORS[op]} {bind(expr['value'])}"
        if op == "CONTAINS":
            return f"{field} LIKE {bind('%' + _like_escape(str(expr['value'])) + '%')}"
        if op == "STARTSWITH":
            return f"{field} LIKE {bind(_like_escape(str(expr['value'])) + '%')}"
        raise ValueError(f"Unknown filter operator: {op}")

    @staticmethod
    def _scalar_text(value: Any) -> str:
        """Render a filter value the way ``->>`` renders the stored JSON scalar."""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _where_clauses(
        self,
        filters: Optional[Union[Dict[str, Any], List[FilterExpr]]],
        user_id: Optional[str],
        params: Dict[str, Any],
    ) -> List[str]:
        clauses: List[str] = []
        if user_id is not None:
            clauses.append("(user_id = :scope_user_id OR user_id IS NULL)")
            params["scope_user_id"] = user_id
        if filters is None:
            return clauses
        if isinstance(filters, dict):
            for key, value in filters.items():
                if isinstance(value, (dict, list)):
                    name = f"f{len(params)}"
                    key_name = f"fk{len(params)}"
                    params[name] = json.dumps(value, default=str)
                    params[key_name] = key
                    clauses.append(f"(meta_data -> :{key_name}) = CAST(:{name} AS jsonb)")
                else:
                    clauses.append(self._filter_to_sql({"op": "EQ", "key": key, "value": value}, params))
        else:
            for f in filters:
                clauses.append(self._filter_to_sql(f.to_dict() if hasattr(f, "to_dict") else f, params))  # type: ignore[arg-type]
        return clauses

    def _distance_sql(self) -> str:
        operator = _DISTANCE_OPERATORS.get(self.distance)
        if operator is None:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        return f"embedding {operator} CAST(:query_embedding AS {self.vector_type})"

    def _recency_sql(self) -> str:
        return "COALESCE(updated_at, created_at)"

    def _with_recency(self, meta_data: Any, timestamp: Any) -> Dict[str, Any]:
        merged = dict(_json_value(meta_data) or {})
        if self.return_updated_at and isinstance(timestamp, datetime):
            merged[STORE_RECENCY_METADATA_KEY] = timestamp.isoformat()
        return merged

    def _to_document(self, row: Any, meta_data: Dict[str, Any]) -> Document:
        return Document(
            id=row.id,
            name=row.name,
            meta_data=meta_data,
            content=row.content,
            embedder=self.embedder,
            usage=_json_value(row.usage),
        )

    def _run_search(self, sql: str, params: Dict[str, Any], vector_search: bool) -> List[Any]:
        with self.db_engine.begin() as conn:
            if vector_search and self.vector_index is not None:
                if isinstance(self.vector_index, Ivfflat):
                    conn.execute(text(f"SET LOCAL ivfflat_probes = {int(self.vector_index.probes)}"))
                elif isinstance(self.vector_index, HNSW):
                    conn.execute(text(f"SET LOCAL hnsw_ef_search = {int(self.vector_index.ef_search)}"))
            return list(conn.execute(text(sql), params).fetchall())

    def _select_columns(self) -> str:
        return f'id, name, meta_data, content, "usage", {self._recency_sql()} AS recency'

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Perform a search based on the configured search type.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Union[Dict[str, Any], List[FilterExpr]]]): Metadata filters.
            user_id (Optional[str]): Restrict results to this user's rows plus shared
                (``user_id IS NULL``) rows. ``None`` applies no owner predicate.
        """
        if self.search_type == SearchType.vector:
            return self.vector_search(query=query, limit=limit, filters=filters, user_id=user_id)
        elif self.search_type == SearchType.keyword:
            return self.keyword_search(query=query, limit=limit, filters=filters, user_id=user_id)
        elif self.search_type == SearchType.hybrid:
            return self.hybrid_search(query=query, limit=limit, filters=filters, user_id=user_id)
        log_error(f"Invalid search type '{self.search_type}'.")
        return []

    async def async_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        return await asyncio.to_thread(self.search, query, limit, filters, user_id)

    def _embed_query(self, query: str) -> Optional[str]:
        query_embedding = self.embedder.get_embedding(query)
        if not query_embedding:
            log_error(f"Error getting embedding for Query: {query}")
            return None
        return _vector_literal(query_embedding)

    def _finish(self, results: List[Document], query: str) -> List[Document]:
        if self.reranker:
            results = self.reranker.rerank(query=query, documents=results)
        log_info(f"Found {len(results)} documents")
        return results

    def vector_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """Perform a vector similarity search."""
        try:
            query_embedding = self._embed_query(query)
            if query_embedding is None:
                return []

            distance_sql = self._distance_sql()
            params: Dict[str, Any] = {"query_embedding": query_embedding, "limit": int(limit)}
            clauses = self._where_clauses(filters, user_id, params)

            if self.similarity_threshold is not None:
                threshold = score_to_distance_threshold(self.similarity_threshold, self.distance)
                if self.distance == Distance.max_inner_product:
                    # ``<#>`` yields the negative inner product
                    threshold = -threshold
                clauses.append(f"{distance_sql} <= :distance_threshold")
                params["distance_threshold"] = threshold

            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = (
                f"SELECT {self._select_columns()}, {distance_sql} AS distance FROM {self.fullname} "
                f"{where} ORDER BY {distance_sql} LIMIT :limit"
            )
            log_debug(f"Vector search query: {sql}")
            rows = self._run_search(sql, params, vector_search=True)

            results: List[Document] = []
            for row in rows:
                raw_distance = -row.distance if self.distance == Distance.max_inner_product else row.distance
                meta_data = self._with_recency(row.meta_data, row.recency)
                meta_data["similarity_score"] = normalize_score(float(raw_distance), self.distance)
                results.append(self._to_document(row, meta_data))
            return self._finish(results, query)
        except EmbeddingError:
            raise
        except Exception as e:
            log_error(f"Error during vector search: {str(e)}")
            return []

    def _tsvector_sql(self) -> str:
        return f"to_tsvector('{self.content_language}', content)"

    def _tsquery_sql(self) -> str:
        return f"plainto_tsquery('{self.content_language}', :query)"

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """Perform a full-text search on the ``content`` column."""
        try:
            params: Dict[str, Any] = {"query": query, "limit": int(limit)}
            clauses = [f"{self._tsvector_sql()} @@ {self._tsquery_sql()}"]
            clauses.extend(self._where_clauses(filters, user_id, params))
            rank = f"ts_rank_cd({self._tsvector_sql()}, {self._tsquery_sql()})"
            sql = (
                f"SELECT {self._select_columns()}, {rank} AS rank FROM {self.fullname} "
                f"WHERE {' AND '.join(clauses)} ORDER BY {rank} DESC LIMIT :limit"
            )
            log_debug(f"Keyword search query: {sql}")
            rows = self._run_search(sql, params, vector_search=False)

            results: List[Document] = []
            for row in rows:
                meta_data = self._with_recency(row.meta_data, row.recency)
                rank_value = float(row.rank or 0.0)
                meta_data["similarity_score"] = rank_value / (rank_value + _RANK_NORMALIZATION_K)
                results.append(self._to_document(row, meta_data))
            return self._finish(results, query)
        except Exception as e:
            log_error(f"Error during keyword search: {str(e)}")
            return []

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Union[Dict[str, Any], List[FilterExpr]]] = None,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """Perform a hybrid search combining vector similarity and full-text rank."""
        try:
            query_embedding = self._embed_query(query)
            if query_embedding is None:
                return []

            distance_sql = self._distance_sql()
            if self.distance == Distance.l2:
                vector_score = f"(1 / (1 + ({distance_sql})))"
            elif self.distance == Distance.cosine:
                vector_score = f"GREATEST(0.0, 1 - ({distance_sql}))"
            elif self.distance == Distance.max_inner_product:
                vector_score = f"GREATEST(0.0, LEAST(1.0, ((-({distance_sql})) + 1) / 2))"
            else:
                log_error(f"Unknown distance metric: {self.distance}")
                return []

            raw_rank = f"ts_rank_cd({self._tsvector_sql()}, {self._tsquery_sql()})"
            text_rank = f"({raw_rank} / ({raw_rank} + {_RANK_NORMALIZATION_K}))"
            text_weight = 1 - self.vector_score_weight
            hybrid_score = f"({self.vector_score_weight} * {vector_score} + {text_weight} * {text_rank})"

            params: Dict[str, Any] = {"query_embedding": query_embedding, "query": query, "limit": int(limit)}
            clauses = self._where_clauses(filters, user_id, params)
            if self.similarity_threshold is not None:
                clauses.append(f"{hybrid_score} >= :score_threshold")
                params["score_threshold"] = self.similarity_threshold

            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = (
                f"SELECT {self._select_columns()}, {hybrid_score} AS hybrid_score FROM {self.fullname} "
                f"{where} ORDER BY {hybrid_score} DESC LIMIT :limit"
            )
            log_debug(f"Hybrid search query: {sql}")
            rows = self._run_search(sql, params, vector_search=True)

            results: List[Document] = []
            for row in rows:
                meta_data = self._with_recency(row.meta_data, row.recency)
                meta_data["similarity_score"] = float(row.hybrid_score)
                results.append(self._to_document(row, meta_data))
            return self._finish(results, query)
        except EmbeddingError:
            raise
        except Exception as e:
            log_error(f"Error during hybrid search: {str(e)}")
            return []

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    def optimize(self, force_recreate: bool = False) -> None:
        """Create (or recreate) the vector index and the full-text index."""
        log_debug("==== Optimizing Vector DB ====")
        self._create_vector_index(force_recreate=force_recreate)
        self._create_text_index(force_recreate=force_recreate)
        log_debug("==== Optimized Vector DB ====")

    async def async_optimize(self, force_recreate: bool = False) -> None:
        await asyncio.to_thread(self.optimize, force_recreate)

    def _index_exists(self, index_name: str) -> bool:
        with self.db_engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM pg_indexes WHERE schemaname = :s AND indexname = :i"),
                {"s": self.schema, "i": index_name},
            ).first()
            return row is not None

    def _drop_index(self, index_name: str) -> None:
        with self.db_engine.begin() as conn:
            conn.execute(text(f'DROP INDEX IF EXISTS "{self.schema}"."{index_name}"'))

    def _prepare_index(self, index_name: str, force_recreate: bool) -> bool:
        """Return True when the index should be (re)created."""
        if not self._index_exists(index_name):
            return True
        if not force_recreate:
            log_info(f"Skipping index creation as index '{index_name}' already exists.")
            return False
        log_info(f"Force recreating index '{index_name}'. Dropping existing index.")
        self._drop_index(index_name)
        return True

    def _create_vector_index(self, force_recreate: bool = False) -> None:
        if self.vector_index is None:
            log_debug("No vector index specified, skipping vector index optimization.")
            return

        index_type = "ivfflat" if isinstance(self.vector_index, Ivfflat) else "hnsw"
        if self.vector_index.name is None:
            self.vector_index.name = f"{self.table_name}_{index_type}_index"
        index_name = _check_identifier(self.vector_index.name, "index name")
        if not self._prepare_index(index_name, force_recreate):
            return

        opclass = f"{self.vector_type}_{_INDEX_OPCLASS_SUFFIX.get(self.distance, 'cosine_ops')}"
        if isinstance(self.vector_index, Ivfflat):
            lists = self.vector_index.lists
            if self.vector_index.dynamic_lists:
                total = self.get_count()
                lists = max(int(total / 1000), 1) if total < 1000000 else max(int(sqrt(total)), 1)
            options = f"lists = {int(lists)}"
        else:
            options = f"m = {int(self.vector_index.m)}, ef_construction = {int(self.vector_index.ef_construction)}"

        sql = (
            f'CREATE INDEX "{index_name}" ON {self.fullname} USING {index_type} (embedding {opclass}) WITH ({options})'
        )
        log_debug(f"Creating vector index: {sql}")
        with self.db_engine.begin() as conn:
            conn.execute(text(sql))

    def _create_text_index(self, force_recreate: bool = False) -> None:
        index_name = f"{self.table_name}_content_gin_index"
        if not self._prepare_index(index_name, force_recreate):
            return
        sql = f'CREATE INDEX "{index_name}" ON {self.fullname} USING GIN ({self._tsvector_sql()})'
        log_debug(f"Creating text index: {sql}")
        with self.db_engine.begin() as conn:
            conn.execute(text(sql))

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def _delete_where(self, where: str, params: Dict[str, Any], description: str) -> bool:
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {self.fullname} WHERE {where}"), params)
            log_info(f"Deleted records with {description} from table '{self.fullname}'.")
            return True
        except Exception as e:
            log_error(f"Error deleting rows from table '{self.fullname}': {str(e)}")
            return False

    def delete(self) -> bool:
        """Delete all records from the table."""
        return self._delete_where("1 = 1", {}, "all rows")

    def delete_by_id(self, id: str) -> bool:
        return self._delete_where("id = :id", {"id": id}, f"id '{id}'")

    def delete_by_name(self, name: str) -> bool:
        return self._delete_where("name = :name", {"name": name}, f"name '{name}'")

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        params: Dict[str, Any] = {}
        clauses = self._where_clauses(metadata, None, params)
        if not clauses:
            log_warning("delete_by_metadata called with empty metadata; nothing was deleted.")
            return False
        return self._delete_where(" AND ".join(clauses), params, f"metadata '{metadata}'")

    def delete_by_content_id(self, content_id: str, user_id: Optional[str] = None) -> bool:
        """Delete content by content ID. Without ``user_id`` the delete spans every owner."""
        where, params = "content_id = :content_id", {"content_id": content_id}
        if user_id is not None:
            where += " AND user_id = :owner_id"
            params["owner_id"] = user_id
        return self._delete_where(where, params, f"content ID '{content_id}'")

    def _delete_by_content_hash(self, content_hash: str, user_id: Optional[str] = None) -> bool:
        where, params = self._owner_clause("content_hash = :content_hash", user_id)
        params["content_hash"] = content_hash
        return self._delete_where(where, params, f"content hash '{content_hash}'")

    def __deepcopy__(self, memo):
        from copy import deepcopy

        cls = self.__class__
        copied_obj = cls.__new__(cls)
        memo[id(self)] = copied_obj
        for k, v in self.__dict__.items():
            if k in {"db_engine", "embedder"}:
                setattr(copied_obj, k, v)
            else:
                setattr(copied_obj, k, deepcopy(v, memo))
        return copied_obj

    def get_supported_search_types(self) -> List[str]:
        return [SearchType.vector, SearchType.keyword, SearchType.hybrid]
