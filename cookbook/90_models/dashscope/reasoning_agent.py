"""
DashScope Reasoning Agent
=========================

Use a Qwen model with native reasoning as the reasoning model of another agent.
The reasoning manager detects DashScope reasoning models automatically (thinking
enabled, or dedicated models such as qwq-* and qvq-*) and streams the reasoning
trace as it is produced.
"""

from agno.agent import Agent
from agno.models.dashscope import DashScope

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

agent = Agent(
    model=DashScope(id="qwen-plus"),
    reasoning_model=DashScope(id="qwen-plus", enable_thinking=True),
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent.print_response(
        "A farmer must cross a river with a fox, a chicken and a sack of grain. "
        "The boat fits the farmer and one item. Give a step-by-step solution.",
        stream=True,
        show_full_reasoning=True,
    )
