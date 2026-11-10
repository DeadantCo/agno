"""Show how a tool hook can access the Function object (tool_definition) of the tool it intercepts.

A hook opts in by declaring a `tool_definition` parameter. Here the hook changes
`stop_after_tool_call` at runtime, so the agent only stops when the decision is approved.
Note that the Function object is shared by every call to the tool, so mutating it
affects concurrent calls to the same tool.
"""

from typing import Any, Callable, Dict

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.tools.function import Function

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------


@tool(stop_after_tool_call=True)
def submit_decision(decision: str) -> str:
    """Submit a final decision.

    Args:
        decision: The decision to submit.
    """
    return f"Decision submitted: {decision}"


def review_hook(
    function_name: str,
    function_call: Callable,
    arguments: Dict[str, Any],
    tool_definition: Function,
):
    if function_name != "submit_decision":
        return function_call(**arguments)

    if "approved" in arguments["decision"].lower():
        # Approved: stop the agent after this tool call
        tool_definition.stop_after_tool_call = True
        return function_call(**arguments)

    # Rejected: keep the agent loop going so it can retry
    tool_definition.stop_after_tool_call = False
    return "REJECTED: the decision must contain the word 'approved'. Please try again."


agent = Agent(
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[submit_decision],
    tool_hooks=[review_hook],
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent.print_response("Submit a decision to launch the product.", stream=True)
