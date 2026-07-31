from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime

load_dotenv()


class SafetyInputMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.guard_model = ChatGroq(model="llama-3.3-70b-versatile")

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:

        if not state["messages"]:
            return None

        last_message = state["messages"][-1]

        prompt = f"""
You are a safety classifier.

Determine whether the following user request is safe.

Respond with exactly one word:
SAFE
or
UNSAFE

User request:
{last_message.content}
"""

        result = self.guard_model.invoke(prompt)

        if result.content.strip().upper() == "UNSAFE":
            state["messages"].append(
                AIMessage(
                    content="Your request was blocked because it violates the safety policy."
                )
            )
            return {"jump_to": "end"}

        return None


@tool
def general_tool(query: str) -> str:
    """General purpose tool."""
    return f"Tool result: {query}"


llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_agent(
    model=llm,
    tools=[general_tool],
    middleware=[SafetyInputMiddleware()],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Nepal?"
            }
        ]
    }
)

print(result["messages"][-1].content)