"""LiteLLM should not send temperature/top_p unless the user sets them."""

from agno.models.litellm import LiteLLM


def test_default_request_params_omit_sampling_params():
    params = LiteLLM(id="anthropic/claude-sonnet-4-5").get_request_params()

    assert "temperature" not in params
    assert "top_p" not in params


def test_explicit_sampling_params_are_sent():
    params = LiteLLM(id="anthropic/claude-sonnet-4-5", temperature=0.3, top_p=0.9).get_request_params()

    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.9
