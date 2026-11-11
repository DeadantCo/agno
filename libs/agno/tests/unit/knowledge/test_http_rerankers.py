import asyncio

import httpx

from agno.knowledge.document import Document
from agno.knowledge.reranker.dashscope import DashScopeReranker
from agno.knowledge.reranker.xinference import XinferenceReranker


def _docs():
    return [Document(content="a"), Document(content="b"), Document(content="c")]


def _patch_post(monkeypatch, payload, captured):
    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)


def _patch_async_post(monkeypatch, payload, captured):
    async def fake_post(self, url, json=None, headers=None, **kwargs):
        captured.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)


DASHSCOPE_PAYLOAD = {
    "output": {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.7},
        ]
    }
}
XINFERENCE_PAYLOAD = {
    "id": "x",
    "results": [
        {"index": 1, "relevance_score": 0.2},
        {"index": 0, "relevance_score": 0.8},
        {"index": 2, "relevance_score": 0.4},
    ],
}


def test_dashscope_rerank_orders_and_scores(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, DASHSCOPE_PAYLOAD, captured)
    reranker = DashScopeReranker(api_key="k", top_n=2, instruct="rank it")

    result = reranker.rerank("q", _docs())

    assert [d.content for d in result] == ["c", "b"]
    assert [d.reranking_score for d in result] == [0.9, 0.7]
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["json"]["model"] == "gte-rerank-v2"
    assert captured["json"]["input"] == {"query": "q", "documents": ["a", "b", "c"]}
    assert captured["json"]["parameters"] == {"return_documents": False, "top_n": 2, "instruct": "rank it"}


def test_dashscope_arerank(monkeypatch):
    captured = {}
    _patch_async_post(monkeypatch, DASHSCOPE_PAYLOAD, captured)
    reranker = DashScopeReranker(api_key="k")

    result = asyncio.run(reranker.arerank("q", _docs()))

    assert [d.content for d in result] == ["c", "b", "a"]


def test_dashscope_missing_api_key_returns_original(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    docs = _docs()
    assert DashScopeReranker().rerank("q", docs) == docs


def test_dashscope_http_error_returns_original(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    docs = _docs()
    assert DashScopeReranker(api_key="k").rerank("q", docs) == docs


def test_dashscope_empty_documents():
    assert DashScopeReranker(api_key="k").rerank("q", []) == []
    assert asyncio.run(DashScopeReranker(api_key="k").arerank("q", [])) == []


def test_dashscope_invalid_top_n_ignored(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, DASHSCOPE_PAYLOAD, captured)
    result = DashScopeReranker(api_key="k", top_n=-1).rerank("q", _docs())
    assert len(result) == 3
    assert "top_n" not in captured["json"]["parameters"]


def test_xinference_rerank_orders_and_scores(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, XINFERENCE_PAYLOAD, captured)
    reranker = XinferenceReranker(model="bge-reranker-v2-m3", base_url="http://host:9997/", top_n=2)

    result = reranker.rerank("q", _docs())

    assert [d.content for d in result] == ["a", "c"]
    assert [d.reranking_score for d in result] == [0.8, 0.4]
    assert captured["url"] == "http://host:9997/v1/rerank"
    assert "Authorization" not in captured["headers"]
    assert captured["json"] == {
        "model": "bge-reranker-v2-m3",
        "query": "q",
        "documents": ["a", "b", "c"],
        "return_documents": False,
        "top_n": 2,
    }


def test_xinference_endpoint_with_v1_suffix_and_api_key(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, XINFERENCE_PAYLOAD, captured)
    XinferenceReranker(base_url="http://host:9997/v1", api_key="secret").rerank("q", _docs())
    assert captured["url"] == "http://host:9997/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_xinference_arerank(monkeypatch):
    captured = {}
    _patch_async_post(monkeypatch, XINFERENCE_PAYLOAD, captured)
    result = asyncio.run(XinferenceReranker().arerank("q", _docs()))
    assert [d.content for d in result] == ["a", "c", "b"]
    assert captured["url"] == "http://localhost:9997/v1/rerank"


def test_xinference_bad_response_returns_original(monkeypatch):
    _patch_post(monkeypatch, {"detail": "nope"}, {})
    docs = _docs()
    assert XinferenceReranker().rerank("q", docs) == docs
