"""Team coordinate mode streams RemoteAgent members with yield_run_output=True.

The AgentOS server then emits the terminal RunOutput as an SSE payload which has
`events` but no `event` key. The client must not treat it as an unknown event.
"""

import json
from typing import List
from unittest.mock import patch

import pytest

from agno.agent.remote import RemoteAgent
from agno.client.os import AgentOSClient
from agno.run.agent import RunOutput


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload)


async def _lines(lines: List[str]):
    for line in lines:
        yield line


@pytest.mark.asyncio
async def test_remote_agent_stream_with_yield_run_output_handles_final_run_output():
    started = {"event": "RunStarted", "run_id": "r1", "agent_id": "a1", "session_id": "s1"}
    content = {"event": "RunContent", "run_id": "r1", "agent_id": "a1", "content": "hi"}
    final_run_output = {
        "run_id": "r1",
        "agent_id": "a1",
        "agent_name": "Agent",
        "session_id": "s1",
        "content": "hi",
        "status": "COMPLETED",
        "events": [started, content],
    }
    lines = [_sse(started), "", _sse(content), "", _sse(final_run_output), ""]

    remote_agent = RemoteAgent(base_url="http://fake-host", agent_id="a1")

    captured = {}

    def fake_stream(self, endpoint, data, headers=None):
        captured["data"] = data
        return _lines(lines)

    with (
        patch.object(AgentOSClient, "_astream_post_form_data", fake_stream),
        patch("agno.utils.log.logger.exception") as log_exc,
    ):
        items = [
            item
            async for item in remote_agent.arun(
                "hello",
                stream=True,
                stream_events=True,
                yield_run_output=True,
            )
        ]

    assert captured["data"].get("yield_run_output") is not None
    # The terminal RunOutput must be surfaced (team delegate code captures it) and
    # must not be logged as an "Unknown event type" error.
    assert not log_exc.called, f"Unexpected error logged: {log_exc.call_args_list}"
    assert any(isinstance(i, RunOutput) for i in items)


@pytest.mark.asyncio
async def test_remote_team_stream_with_yield_run_output_handles_final_run_output():
    from agno.run.team import TeamRunOutput

    started = {"event": "TeamRunStarted", "run_id": "r1", "team_id": "t1", "session_id": "s1"}
    final_run_output = {
        "run_id": "r1",
        "team_id": "t1",
        "team_name": "Team",
        "session_id": "s1",
        "content": "hi",
        "status": "COMPLETED",
        "events": [started],
    }
    lines = [_sse(started), "", _sse(final_run_output), ""]
    client = AgentOSClient(base_url="http://fake-host")

    def fake_stream(self, endpoint, data, headers=None):
        return _lines(lines)

    with (
        patch.object(AgentOSClient, "_astream_post_form_data", fake_stream),
        patch("agno.utils.log.logger.exception") as log_exc,
    ):
        items = [item async for item in client.run_team_stream(team_id="t1", message="hello", yield_run_output=True)]

    assert not log_exc.called, f"Unexpected error logged: {log_exc.call_args_list}"
    assert isinstance(items[-1], TeamRunOutput)
    assert len(items) == 2
