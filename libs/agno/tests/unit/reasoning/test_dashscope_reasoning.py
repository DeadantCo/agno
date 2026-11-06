"""Unit tests for DashScope Qwen native reasoning detection, routing and streaming."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agno.models.dashscope import DashScope
from agno.models.message import Message
from agno.reasoning.dashscope import (
    aget_dashscope_reasoning,
    aget_dashscope_reasoning_stream,
    get_dashscope_reasoning,
    get_dashscope_reasoning_stream,
    is_dashscope_reasoning_model,
)
from agno.reasoning.manager import ReasoningManager
from agno.run.agent import RunEvent


def _model(model_id, **kwargs):
    return DashScope(id=model_id, api_key="test", **kwargs)


@pytest.mark.parametrize(
    "model_id,kwargs,expected",
    [
        ("qwen-plus", {"enable_thinking": True}, True),
        ("qwen3-max", {"include_thoughts": True}, True),
        ("qwq-plus", {}, True),
        ("qvq-max", {}, True),
        ("qwen3-235b-a22b-thinking-2507", {}, True),
        ("qwen-plus", {}, False),
        ("qwen-plus", {"enable_thinking": True, "include_thoughts": False}, False),
        ("deepseek-v4-pro", {}, False),
    ],
)
def test_is_dashscope_reasoning_model(model_id, kwargs, expected):
    assert is_dashscope_reasoning_model(_model(model_id, **kwargs)) is expected


def test_non_dashscope_model_is_not_matched():
    from agno.models.openai.like import OpenAILike

    assert is_dashscope_reasoning_model(OpenAILike(id="qwq-plus", api_key="x")) is False


def test_manager_routes_dashscope_before_openai():
    manager = ReasoningManager.__new__(ReasoningManager)
    assert manager._detect_model_type_uncached(_model("qwen-plus", enable_thinking=True)) == "dashscope"
    assert manager._detect_model_type_uncached(_model("qwq-plus")) == "dashscope"


def test_manager_keeps_deepseek_via_dashscope_on_openai_route():
    manager = ReasoningManager.__new__(ReasoningManager)
    assert manager._detect_model_type_uncached(_model("deepseek-v4-pro")) == "openai"


def _event(event_type, **kwargs):
    return SimpleNamespace(event=event_type, **kwargs)


def _events():
    return [
        _event(RunEvent.run_content, reasoning_content="step 1 "),
        _event(RunEvent.run_content, reasoning_content=None, content="answer"),
        _event(RunEvent.run_content, reasoning_content="step 2"),
        _event(RunEvent.run_completed, metrics=None),
    ]


def _sync_agent():
    agent = MagicMock()
    agent.run.return_value = iter(_events())
    return agent


def _async_agent():
    async def arun(**kwargs):
        for event in _events():
            yield event

    agent = MagicMock()
    agent.arun.side_effect = lambda **kwargs: arun(**kwargs)
    return agent


def test_sync_reasoning_aggregates_streamed_reasoning():
    agent = _sync_agent()
    messages = [Message(role="developer", content="hi")]

    result = get_dashscope_reasoning(agent, messages)

    assert result is not None
    assert result.reasoning_content == "step 1 step 2"
    assert result.content == "<thinking>\nstep 1 step 2\n</thinking>"
    assert messages[0].role == "system"
    assert agent.run.call_args.kwargs["stream"] is True


def test_sync_reasoning_error_returns_none():
    agent = MagicMock()
    agent.run.side_effect = RuntimeError("boom")
    assert get_dashscope_reasoning(agent, [Message(role="user", content="hi")]) is None


def test_sync_stream_yields_deltas_then_final_message():
    items = list(get_dashscope_reasoning_stream(_sync_agent(), [Message(role="user", content="hi")]))

    assert [d for d, _ in items[:-1]] == ["step 1 ", "step 2"]
    assert all(m is None for _, m in items[:-1])
    assert items[-1][0] is None
    assert items[-1][1].reasoning_content == "step 1 step 2"


def test_sync_stream_without_reasoning_yields_nothing():
    agent = MagicMock()
    agent.run.return_value = iter([_event(RunEvent.run_content, reasoning_content=None, content="x")])
    assert list(get_dashscope_reasoning_stream(agent, [Message(role="user", content="hi")])) == []


def test_async_reasoning_aggregates_streamed_reasoning():
    result = asyncio.run(aget_dashscope_reasoning(_async_agent(), [Message(role="user", content="hi")]))
    assert result is not None
    assert result.reasoning_content == "step 1 step 2"


def test_async_stream_yields_deltas_then_final_message():
    async def collect():
        return [x async for x in aget_dashscope_reasoning_stream(_async_agent(), [Message(role="user", content="hi")])]

    items = asyncio.run(collect())
    assert [d for d, _ in items[:-1]] == ["step 1 ", "step 2"]
    assert items[-1][1].reasoning_content == "step 1 step 2"
