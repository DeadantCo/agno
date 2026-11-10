"""Tests that LiteLLM only sends sampling params when they are set on the model."""

from agno.models.litellm import LiteLLM


def test_sampling_params_unset_by_default():
    model = LiteLLM(id="anthropic/claude-sonnet-4-5")

    assert model.temperature is None
    assert model.top_p is None
    assert model.get_request_params() == {"model": "anthropic/claude-sonnet-4-5"}


def test_sampling_params_sent_when_set():
    model = LiteLLM(id="anthropic/claude-sonnet-4-5", temperature=0.7, top_p=1.0)

    params = model.get_request_params()

    assert params["temperature"] == 0.7
    assert params["top_p"] == 1.0


def test_zero_temperature_is_sent():
    model = LiteLLM(id="anthropic/claude-sonnet-4-5", temperature=0.0)

    params = model.get_request_params()

    assert params["temperature"] == 0.0
    assert "top_p" not in params


def test_only_set_sampling_param_is_sent():
    model = LiteLLM(id="anthropic/claude-sonnet-4-5", top_p=0.9)

    params = model.get_request_params()

    assert params["top_p"] == 0.9
    assert "temperature" not in params
