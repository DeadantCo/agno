import pytest

from agno.tools.function import Function, FunctionCall


def _make(hook):
    def submit(x: int) -> str:
        return "ok"

    f = Function.from_callable(submit)
    f.tool_hooks = [hook]
    f.stop_after_tool_call = True
    return f, FunctionCall(function=f, arguments={"x": 1}, call_id="1")


def test_sync_hook_receives_tool_definition():
    seen = {}

    def hook(function_name, function_call, arguments, tool_definition):
        seen["td"] = tool_definition
        return function_call(**arguments)

    f, fc = _make(hook)
    fc.execute()
    assert seen.get("td") is f


@pytest.mark.asyncio
async def test_async_hook_receives_tool_definition():
    seen = {}

    async def hook(function_name, function_call, arguments, tool_definition):
        seen["td"] = tool_definition
        return await function_call(**arguments)

    f, fc = _make(hook)
    await fc.aexecute()
    assert seen.get("td") is f


def test_sync_hook_can_override_stop_after_tool_call():
    def hook(function_name, function_call, arguments, tool_definition):
        tool_definition.stop_after_tool_call = False
        return function_call(**arguments)

    f, fc = _make(hook)
    fc.execute()
    assert f.stop_after_tool_call is False


@pytest.mark.asyncio
async def test_async_hook_can_override_stop_after_tool_call():
    async def hook(function_name, function_call, arguments, tool_definition):
        tool_definition.stop_after_tool_call = False
        return await function_call(**arguments)

    f, fc = _make(hook)
    await fc.aexecute()
    assert f.stop_after_tool_call is False


def test_hook_without_tool_definition_is_unaffected():
    def hook(function_name, function_call, arguments):
        return function_call(**arguments)

    f, fc = _make(hook)
    assert fc.execute().result == "ok"
    assert f.stop_after_tool_call is True
