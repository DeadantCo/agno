from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Iterator, List, Optional, Tuple

from agno.models.base import Model
from agno.models.message import Message
from agno.utils.log import log_warning

if TYPE_CHECKING:
    from agno.metrics import RunMetrics

# Model ids that always reason, regardless of the enable_thinking flag.
_DASHSCOPE_ALWAYS_REASONING_SUBSTRINGS = ("qwq", "qvq", "-thinking")


def is_dashscope_reasoning_model(reasoning_model: Model) -> bool:
    """Check if a DashScope model runs with Qwen native reasoning.

    Matches DashScope models that have thinking switched on (``enable_thinking`` or
    ``include_thoughts``), and the dedicated reasoning models that always think
    (qwq-*, qvq-*, *-thinking-*).
    """
    if reasoning_model.__class__.__name__ != "DashScope":
        return False

    include_thoughts = getattr(reasoning_model, "include_thoughts", None)
    enable_thinking = getattr(reasoning_model, "enable_thinking", False)
    if include_thoughts is not None:
        enable_thinking = include_thoughts
    if enable_thinking:
        return True

    model_id = reasoning_model.id.lower()
    return any(substring in model_id for substring in _DASHSCOPE_ALWAYS_REASONING_SUBSTRINGS)


def _prepare_messages(messages: List[Message]) -> None:
    for message in messages:
        if message.role == "developer":
            message.role = "system"


def _build_message(reasoning_content: str) -> Message:
    return Message(
        role="assistant", content=f"<thinking>\n{reasoning_content}\n</thinking>", reasoning_content=reasoning_content
    )


def _accumulate_metrics(completed_event, run_metrics: Optional["RunMetrics"]) -> None:
    if run_metrics is not None:
        from agno.metrics import accumulate_eval_metrics

        accumulate_eval_metrics(completed_event.metrics, run_metrics, prefix="reasoning")


def get_dashscope_reasoning(
    reasoning_agent: "Agent",  # type: ignore[name-defined]  # noqa: F821
    messages: List[Message],
    run_metrics: Optional["RunMetrics"] = None,
) -> Optional[Message]:
    """Collect Qwen reasoning content. The model is always called in streaming mode because
    DashScope only returns the reasoning trace for streaming requests."""
    from agno.run.agent import RunEvent

    _prepare_messages(messages)

    reasoning_content: str = ""
    try:
        for event in reasoning_agent.run(input=messages, stream=True, stream_events=True):
            event_type = getattr(event, "event", None)
            if event_type == RunEvent.run_content:
                delta = getattr(event, "reasoning_content", None)
                if delta:
                    reasoning_content += delta
            elif event_type == RunEvent.run_completed:
                _accumulate_metrics(event, run_metrics)
    except Exception as e:
        log_warning(f"Reasoning error: {str(e)}")
        return None

    return _build_message(reasoning_content)


async def aget_dashscope_reasoning(
    reasoning_agent: "Agent",  # type: ignore[name-defined]  # noqa: F821
    messages: List[Message],
    run_metrics: Optional["RunMetrics"] = None,
) -> Optional[Message]:
    from agno.run.agent import RunEvent

    _prepare_messages(messages)

    reasoning_content: str = ""
    try:
        async for event in reasoning_agent.arun(input=messages, stream=True, stream_events=True):
            event_type = getattr(event, "event", None)
            if event_type == RunEvent.run_content:
                delta = getattr(event, "reasoning_content", None)
                if delta:
                    reasoning_content += delta
            elif event_type == RunEvent.run_completed:
                _accumulate_metrics(event, run_metrics)
    except Exception as e:
        log_warning(f"Reasoning error: {str(e)}")
        return None

    return _build_message(reasoning_content)


def get_dashscope_reasoning_stream(
    reasoning_agent: "Agent",  # type: ignore  # noqa: F821
    messages: List[Message],
) -> Iterator[Tuple[Optional[str], Optional[Message]]]:
    """
    Stream reasoning content from a DashScope Qwen model.

    Yields:
        Tuple of (reasoning_content_delta, final_message)
        - During streaming: (reasoning_content_delta, None)
        - At the end: (None, final_message)
    """
    from agno.run.agent import RunEvent

    _prepare_messages(messages)

    reasoning_content: str = ""

    try:
        for event in reasoning_agent.run(input=messages, stream=True, stream_events=True):
            if getattr(event, "event", None) == RunEvent.run_content:
                delta = getattr(event, "reasoning_content", None)
                if delta:
                    reasoning_content += delta
                    yield (delta, None)
    except Exception as e:
        log_warning(f"Reasoning error: {str(e)}")
        return

    if reasoning_content:
        yield (None, _build_message(reasoning_content))


async def aget_dashscope_reasoning_stream(
    reasoning_agent: "Agent",  # type: ignore  # noqa: F821
    messages: List[Message],
) -> AsyncIterator[Tuple[Optional[str], Optional[Message]]]:
    """
    Stream reasoning content from a DashScope Qwen model asynchronously.

    Yields:
        Tuple of (reasoning_content_delta, final_message)
        - During streaming: (reasoning_content_delta, None)
        - At the end: (None, final_message)
    """
    from agno.run.agent import RunEvent

    _prepare_messages(messages)

    reasoning_content: str = ""

    try:
        async for event in reasoning_agent.arun(input=messages, stream=True, stream_events=True):
            if getattr(event, "event", None) == RunEvent.run_content:
                delta = getattr(event, "reasoning_content", None)
                if delta:
                    reasoning_content += delta
                    yield (delta, None)
    except Exception as e:
        log_warning(f"Reasoning error: {str(e)}")
        return

    if reasoning_content:
        yield (None, _build_message(reasoning_content))
