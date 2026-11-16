"""
openGauss DataVec: Vector Search
====================================
openGauss DataVec adds vector similarity search to openGauss. The OpenGauss
vector db issues openGauss-compatible SQL (no pgvector package, no
CREATE EXTENSION, no INSERT ... ON CONFLICT).

Features:
- Vector, keyword, and hybrid search
- HNSW and IVFFlat indexing
- Per-user isolation through user_id

Setup: run an openGauss instance with DataVec enabled and create a database.
Requires: pip install sqlalchemy psycopg2-binary openai
"""

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIResponses
from agno.vectordb.opengauss import HNSW, OpenGauss
from agno.vectordb.search import SearchType

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

db_url = "postgresql+psycopg2://gaussdb:Gauss_234@localhost:5432/postgres"

knowledge = Knowledge(
    vector_db=OpenGauss(
        table_name="opengauss_recipes",
        schema="ai",
        db_url=db_url,
        search_type=SearchType.hybrid,
        vector_index=HNSW(m=16, ef_construction=200),
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# ---------------------------------------------------------------------------
# Run Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    knowledge.insert(url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf")

    # Build the HNSW and full-text indexes once the table holds data
    knowledge.vector_db.optimize()  # type: ignore[union-attr]

    agent = Agent(
        model=OpenAIResponses(id="gpt-5.2"),
        knowledge=knowledge,
        search_knowledge=True,
        markdown=True,
    )
    agent.print_response(
        "How do I make chicken and galangal in coconut milk soup?", stream=True
    )
