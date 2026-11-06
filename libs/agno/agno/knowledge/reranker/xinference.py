from typing import Any, Dict, List, Optional

import httpx

from agno.knowledge.document import Document
from agno.knowledge.reranker.base import Reranker
from agno.utils.log import logger


class XinferenceReranker(Reranker):
    """Reranker backed by a Xinference server's rerank endpoint (POST {base_url}/v1/rerank)."""

    model: str = "bge-reranker-base"
    base_url: str = "http://localhost:9997"
    api_key: Optional[str] = None
    top_n: Optional[int] = None
    timeout: float = 30.0

    @property
    def endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return f"{base_url}/rerank"
        return f"{base_url}/v1/rerank"

    def _get_top_n(self) -> Optional[int]:
        if self.top_n is not None and self.top_n <= 0:
            logger.warning(f"top_n should be a positive integer, got {self.top_n}, setting top_n to None")
            return None
        return self.top_n

    def _build_request(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [doc.content for doc in documents],
            "return_documents": False,
        }
        top_n = self._get_top_n()
        if top_n:
            body["top_n"] = top_n
        return body

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _process_response(self, data: Dict[str, Any], documents: List[Document]) -> List[Document]:
        results = data.get("results")
        if results is None:
            raise ValueError(f"Unexpected Xinference rerank response: {data}")

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
            self.endpoint, json=self._build_request(query, documents), headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        return self._process_response(response.json(), documents)

    async def _arerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint, json=self._build_request(query, documents), headers=self._headers()
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
