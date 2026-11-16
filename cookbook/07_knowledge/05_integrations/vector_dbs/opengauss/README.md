# openGauss DataVec

Use [openGauss](https://opengauss.org) DataVec as the vector store behind `Knowledge`.

`agno.vectordb.opengauss.OpenGauss` is a standalone adapter. It does not subclass `PgVector`
and does not need the `pgvector` Python package. It uses openGauss-compatible SQL:

- The vector column is created with the DataVec type (`floatvector(<dimensions>)` by default; change it with `vector_type`).
- No `CREATE EXTENSION`.
- Upserts delete and re-insert inside one transaction instead of `INSERT ... ON CONFLICT`.
- Distance operators `<->` (L2), `<=>` (cosine) and `<#>` (inner product); HNSW and IVFFlat indexes through `optimize()`.
- Keyword search uses `plainto_tsquery`.

## Setup

1. Start an openGauss server with DataVec and create a database.
2. Install the dependencies: `pip install sqlalchemy psycopg2-binary openai` (or `pip install "agno[opengauss]"`).
3. Set `OPENAI_API_KEY`.
4. Point `db_url` at your server. Pass `db_engine=` instead if you need a custom SQLAlchemy engine or dialect for openGauss.

## Examples

| File | Description |
| --- | --- |
| `01_opengauss.py` | Hybrid search with an HNSW index and an agent that searches the knowledge base |

Run:

```bash
.venvs/demo/bin/python cookbook/07_knowledge/05_integrations/vector_dbs/opengauss/01_opengauss.py
```
