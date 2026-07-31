from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
load_dotenv()


class HealthcareSafetyFilter(AgentMiddleware):

    BLOCKED_WORDS = [
        "drug synthesis",
        "weapon",
        "hack",
        "suicide",
        "self-harm"
    ]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        if not state["messages"]:
            return None

        message = state["messages"][-1]

        if message.type != "human":
            return None

        text = message.content.lower()

        for word in self.BLOCKED_WORDS:
            if word in text:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "Sorry, I can only help with healthcare-related questions."
                    }],
                    "jump_to": "end",
                }

        return None


class MedicalDisclaimer(AgentMiddleware):

    @hook_config()
    def after_agent(self, state: AgentState, runtime: Runtime):
        last = state["messages"][-1]

        if isinstance(last, AIMessage):
            last.content += (
                "\n\n⚕️ This is general health information. "
                "Please consult a qualified doctor for medical advice."
            )

        return None


@tool
def search_symptoms(symptoms: str) -> str:
    """Search medical symptoms."""
    return f"Possible information about {symptoms}. Please consult a doctor."


@tool
def medication_info(name: str) -> str:
    """Medication information."""
    return f"General information about {name}. Follow your doctor's prescription."


llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_agent(
    model=llm,
    tools=[search_symptoms, medication_info],
    middleware=[
        HealthcareSafetyFilter(),
        MedicalDisclaimer(),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "You are a healthcare assistant. "
        "Answer health questions politely and recommend consulting a doctor."
    ),
)

config = {
    "configurable": {
        "thread_id": "health_chat"
    }
}

print("🏥 Healthcare Assistant")
print("Type 'exit' to quit.\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Goodbye!")
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user
                }
            ]
        },
        config=config,
    )

    print("\nAssistant:", response["messages"][-1].content)
    print("-" * 60)