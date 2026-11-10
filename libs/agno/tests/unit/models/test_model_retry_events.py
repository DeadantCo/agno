import os
from typing import Any, Iterator
from unittest.mock import patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")

from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.metrics import RunMetrics
from agno.models.openai.chat import OpenAIChat
from agno.models.response import ModelResponse, ModelResponseEvent
from agno.run.agent import ModelRetryEvent, RunEvent, RunOutput


def _model(retries: int = 2) -> OpenAIChat:
    return OpenAIChat(id="gpt-4o-mini", retries=retries, delay_between_retries=0)


def _run_response() -> RunOutput:
    return RunOutput(run_id="r1", session_id="s1", agent_id="a1", metrics=RunMetrics())


def _flaky(failures: int, result: Any):
    calls = {"n": 0}

    def fn(**kwargs):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise ModelProviderError("transient", status_code=500)
        return result

    return fn, calls


def test_sync_retries_are_counted_in_run_metrics():
    model = _model()
    run_response = _run_response()
    fn, calls = _flaky(2, ModelResponse(content="ok"))
    with patch.object(model, "invoke", side_effect=fn):
        model._invoke_with_retry(messages=[], run_response=run_response)
    assert calls["n"] == 3
    assert run_response.metrics.retries == 2


@pytest.mark.asyncio
async def test_async_retries_are_counted_in_run_metrics():
    model = _model()
    run_response = _run_response()
    calls = {"n": 0}

    async def fn(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ModelProviderError("transient", status_code=500)
        return ModelResponse(content="ok")

    with patch.object(model, "ainvoke", side_effect=fn):
        await model._ainvoke_with_retry(messages=[], run_response=run_response)
    assert run_response.metrics.retries == 2


def test_no_retry_leaves_metrics_untouched():
    model = _model()
    run_response = _run_response()
    with patch.object(model, "invoke", return_value=ModelResponse(content="ok")):
        model._invoke_with_retry(messages=[], run_response=run_response)
    assert run_response.metrics.retries == 0
    assert "retries" not in run_response.metrics.to_dict()


def test_non_retryable_error_is_not_counted():
    model = _model()
    run_response = _run_response()
    with patch.object(model, "invoke", side_effect=ModelProviderError("bad", status_code=401)):
        with pytest.raises(ModelProviderError):
            model._invoke_with_retry(messages=[], run_response=run_response)
    assert run_response.metrics.retries == 0


def test_stream_retry_yields_retry_events():
    model = _model()
    run_response = _run_response()
    calls = {"n": 0}

    def fn(**kwargs) -> Iterator[ModelResponse]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ModelProviderError("transient", status_code=503)
        yield ModelResponse(content="ok")

    with patch.object(model, "invoke_stream", side_effect=fn):
        out = list(model._invoke_stream_with_retry(messages=[], run_response=run_response))

    retry_events = [r for r in out if r.event == ModelResponseEvent.model_retry.value]
    assert [r.retry_attempt for r in retry_events] == [1, 2]
    assert all(r.max_retries == 2 for r in retry_events)
    assert retry_events[0].retry_error == "transient"
    assert out[-1].content == "ok"
    assert run_response.metrics.retries == 2


@pytest.mark.asyncio
async def test_async_stream_retry_yields_retry_events():
    model = _model()
    run_response = _run_response()
    calls = {"n": 0}

    async def fn(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ModelProviderError("transient", status_code=503)
        yield ModelResponse(content="ok")

    with patch.object(model, "ainvoke_stream", side_effect=fn):
        out = [r async for r in model._ainvoke_stream_with_retry(messages=[], run_response=run_response)]

    retry_events = [r for r in out if r.event == ModelResponseEvent.model_retry.value]
    assert len(retry_events) == 1
    assert retry_events[0].retry_attempt == 1
    assert run_response.metrics.retries == 1


def test_run_metrics_retries_round_trip_and_add():
    a = RunMetrics(retries=2)
    b = RunMetrics(retries=1)
    assert (a + b).retries == 3
    assert RunMetrics.from_dict(a.to_dict()).retries == 2


def _streaming_agent(model: OpenAIChat) -> Agent:
    return Agent(model=model, store_events=True, telemetry=False)


def test_agent_stream_emits_and_stores_model_retry_event():
    model = _model(retries=1)
    calls = {"n": 0}

    def fn(**kwargs) -> Iterator[ModelResponse]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelProviderError("transient", status_code=500)
        yield ModelResponse(content="hello")

    agent = _streaming_agent(model)
    with patch.object(model, "invoke_stream", side_effect=fn):
        events = list(agent.run("hi", stream=True, stream_events=True, yield_run_output=True))

    retry_events = [e for e in events if isinstance(e, ModelRetryEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].event == RunEvent.model_retry.value
    assert retry_events[0].retry_attempt == 1
    assert retry_events[0].model == "gpt-4o-mini"

    run_output = next(e for e in events if isinstance(e, RunOutput))
    assert run_output.metrics is not None
    assert run_output.metrics.retries == 1
    assert any(isinstance(e, ModelRetryEvent) for e in (run_output.events or []))


@pytest.mark.asyncio
async def test_agent_async_stream_emits_model_retry_event():
    model = _model(retries=1)
    calls = {"n": 0}

    async def fn(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelProviderError("transient", status_code=500)
        yield ModelResponse(content="hello")

    agent = _streaming_agent(model)
    with patch.object(model, "ainvoke_stream", side_effect=fn):
        events = [e async for e in agent.arun("hi", stream=True, stream_events=True)]

    retry_events = [e for e in events if isinstance(e, ModelRetryEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].retry_attempt == 1


def test_agent_non_stream_run_reports_retries_in_metrics():
    model = _model(retries=1)
    fn, calls = _flaky(1, ModelResponse(content="hello"))
    agent = _streaming_agent(model)
    with patch.object(model, "invoke", side_effect=fn):
        result = agent.run("hi")
    assert calls["n"] == 2
    assert result.metrics is not None
    assert result.metrics.retries == 1


def test_team_stream_emits_model_retry_event():
    from agno.run.team import ModelRetryEvent as TeamModelRetryEvent
    from agno.run.team import TeamRunEvent
    from agno.team import Team

    model = _model(retries=1)
    calls = {"n": 0}

    def fn(**kwargs) -> Iterator[ModelResponse]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelProviderError("transient", status_code=500)
        yield ModelResponse(content="hello")

    team = Team(members=[Agent(model=_model(), telemetry=False)], model=model, store_events=True, telemetry=False)
    with patch.object(model, "invoke_stream", side_effect=fn):
        events = list(team.run("hi", stream=True, stream_events=True))

    retry_events = [e for e in events if isinstance(e, TeamModelRetryEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].event == TeamRunEvent.model_retry.value
    assert retry_events[0].retry_attempt == 1
