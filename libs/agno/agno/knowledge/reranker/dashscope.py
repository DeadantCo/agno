from os import getenv
from typing import Any, Dict, List, Optional

import httpx

from agno.knowledge.document import Document
from agno.knowledge.reranker.base import Reranker
from agno.utils.log import logger


class DashScopeReranker(Reranker):
    """Reranker backed by the DashScope text-rerank API (e.g. gte-rerank-v2, qwen3-rerank)."""

    model: str = "gte-rerank-v2"
    api_key: Optional[str] = None
    base_url: str = "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    top_n: Optional[int] = None
    instruct: Optional[str] = None
    timeout: float = 30.0

    def _get_api_key(self) -> str:
        api_key = self.api_key or getenv("DASHSCOPE_API_KEY") or getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set. Please set the DASHSCOPE_API_KEY environment variable.")
        return api_key

    def _get_top_n(self) -> Optional[int]:
        if self.top_n is not None and self.top_n <= 0:
            logger.warning(f"top_n should be a positive integer, got {self.top_n}, setting top_n to None")
            return None
        return self.top_n

    def _build_request(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {"return_documents": False}
        top_n = self._get_top_n()
        if top_n:
            parameters["top_n"] = top_n
        if self.instruct:
            parameters["instruct"] = self.instruct
        return {
            "model": self.model,
            "input": {"query": query, "documents": [doc.content for doc in documents]},
            "parameters": parameters,
        }

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._get_api_key()}", "Content-Type": "application/json"}

    def _process_response(self, data: Dict[str, Any], documents: List[Document]) -> List[Document]:
        results = (data.get("output") or {}).get("results")
        if results is None:
            raise ValueError(f"Unexpected DashScope rerank response: {data}")

        reranked: List[Document] = []
        for result in results:
            doc = documents[result["index"]]
            doc.reranking_score = result["relevance_score"]
            reranked.append(doc)

        reranked.sort(
            key=lambda x: x.reranking_score if x.reranking_score is not None else float("-inf"),
            reverse=True,
        )

        top_n = self._get_top_n()
        if top_n:
            reranked = reranked[:top_n]
        return reranked

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        response = httpx.post(
            self.base_url, json=self._build_request(query, documents), headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        return self._process_response(response.json(), documents)

    async def _arerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.base_url, json=self._build_request(query, documents), headers=self._headers()
            )
        response.raise_for_status()
        return self._process_response(response.json(), documents)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            return self._rerank(query=query, documents=documents)
        except Exception:
            logger.exception("Error reranking documents. Returning original documents")
            return documents

    async def arerank(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            return await self._arerank(query=query, documents=documents)
        except Exception:
            logger.exception("Error reranking documents. Returning original documents")
            return documents
