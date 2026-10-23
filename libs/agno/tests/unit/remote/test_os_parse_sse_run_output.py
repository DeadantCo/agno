"""Recreation for #7517: RunOutput payload (no ``event`` key) in the SSE stream
causes ``ValueError: Unknown event type`` in AgentOSClient._parse_sse_events."""

import json
from unittest.mock import patch

import pytest

from agno.client.os import AgentOSClient
from agno.run.agent import RunOutput, run_output_event_from_dict


def test_run_output_dict_is_parseable_as_event():
    payload = RunOutput(run_id="r1", agent_id="a1", agent_name="remote", content="hi").to_dict()
    assert "event" not in payload
    # Expected: a RemoteAgent streaming with yield_run_output=True can handle the final RunOutput
    run_output_event_from_dict(payload)


@pytest.mark.asyncio
async def test_parse_sse_events_does_not_log_unknown_event_for_run_output():
    payload = RunOutput(run_id="r1", agent_id="a1", agent_name="remote", content="hi").to_dict()

    async def raw_stream():
        yield "data: " + json.dumps(payload)

    client = AgentOSClient(base_url="http://localhost:7777")
    with patch("agno.utils.log.logger.exception") as log_exc:
        results = [e async for e in client._parse_sse_events(raw_stream(), run_output_event_from_dict)]
    assert not log_exc.called, log_exc.call_args
    assert len(results) == 1


def test_team_run_output_dict_is_parseable_as_event():
    from agno.run.team import TeamRunOutput, team_run_output_event_from_dict

    payload = TeamRunOutput(run_id="r1", team_id="t1", team_name="team", content="hi").to_dict()
    assert "event" not in payload
    result = team_run_output_event_from_dict(payload)
    assert isinstance(result, TeamRunOutput)
    assert result.content == "hi"

    agent_payload = RunOutput(run_id="r2", agent_id="a1", content="yo").to_dict()
    assert isinstance(team_run_output_event_from_dict(agent_payload), RunOutput)


def test_run_output_dict_roundtrips_to_run_output():
    payload = RunOutput(run_id="r1", agent_id="a1", agent_name="remote", content="hi").to_dict()
    result = run_output_event_from_dict(payload)
    assert isinstance(result, RunOutput)
    assert result.run_id == "r1"
    assert result.content == "hi"


def test_unknown_event_without_run_id_still_raises():
    with pytest.raises(ValueError):
        run_output_event_from_dict({"event": "Nope"})
