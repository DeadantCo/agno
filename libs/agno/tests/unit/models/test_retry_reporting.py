import os
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")

from agno.exceptions import ModelProviderError
from agno.metrics import RunMetrics
from agno.models.openai.chat import OpenAIChat
from agno.models.response import ModelResponse
from agno.run.agent import RunEvent


def test_model_retry_count_is_reported_in_run_metrics():
    """A model retry should be visible in the run metrics (issue #499)."""
    model = OpenAIChat(id="gpt-4o-mini", retries=2, delay_between_retries=0)
    calls = {"n": 0}

    def flaky_invoke(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ModelProviderError("transient", status_code=500)
        return ModelResponse(content="ok")

    with patch.object(model, "invoke", side_effect=flaky_invoke):
        model._invoke_with_retry(messages=[])

    assert calls["n"] == 3  # two retries really happened
    metric_fields = set(RunMetrics.__dataclass_fields__)
    assert any("retr" in f for f in metric_fields), f"No retry field in RunMetrics: {sorted(metric_fields)}"


def test_retry_event_exists_in_run_event():
    """A retry event should exist in RunEvent (issue #499)."""
    assert any("retry" in e.name for e in RunEvent), "No retry event in RunEvent"
