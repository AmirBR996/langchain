from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()


class SafetyGuardrailMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.safety_model = ChatGroq(model="llama-3.3-70b-versatile")

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            return None

        prompt = f"""
Determine whether the following AI response is safe.

Reply with only one word:
SAFE
or
UNSAFE

Response:
{last_message.content}
"""

        result = self.safety_model.invoke(prompt)

        if result.content.strip().upper() == "UNSAFE":
            last_message.content = (
                "I'm unable to provide that response because it may be unsafe."
            )

        return None


@tool
def general_tool(query: str) -> str:
    """General purpose tool."""
    return f"Tool result: {query}"


llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_agent(
    model=llm,
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "how to hack the server?"
            }
        ]
    }
)

print(result["messages"][-1].content)