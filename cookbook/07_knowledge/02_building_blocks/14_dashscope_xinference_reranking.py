"""
DashScope and Xinference Reranking
==================================
Rerank search results with a hosted DashScope model or a self-hosted Xinference
server. Both rerankers call the provider's rerank HTTP API and have no extra
dependencies.

- DashScopeReranker: set DASHSCOPE_API_KEY (or QWEN_API_KEY)
- XinferenceReranker: run a Xinference server with a rerank model launched,
  e.g. `xinference launch --model-name bge-reranker-v2-m3 --model-type rerank`

See also: 07_knowledge_level_reranking.py for how Knowledge-level reranking works.
"""

import asyncio

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reranker.dashscope import DashScopeReranker
from agno.knowledge.reranker.xinference import XinferenceReranker
from agno.models.openai import OpenAIResponses
from agno.vectordb.qdrant import Qdrant

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

qdrant_url = "http://localhost:6333"

dashscope_reranker = DashScopeReranker(model="gte-rerank-v2", top_n=5)
xinference_reranker = XinferenceReranker(
    model="bge-reranker-v2-m3", base_url="http://localhost:9997", top_n=5
)

# Swap in xinference_reranker to use a self-hosted server instead.
knowledge = Knowledge(
    vector_db=Qdrant(collection="dashscope_reranking_demo", url=qdrant_url),
    reranker=dashscope_reranker,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.6-luna"),
    knowledge=knowledge,
    search_knowledge=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    async def main():
        await knowledge.ainsert(
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
        )
        results = await knowledge.asearch(
            "What are some Thai curry dishes?", max_results=5
        )
        print("Reranked results:")
        for document in results:
            print(document.reranking_score, (document.content or "")[:80])

        agent.print_response("What are some Thai curry dishes?", stream=True)

    asyncio.run(main())
