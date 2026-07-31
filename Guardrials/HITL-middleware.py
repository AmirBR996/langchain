from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'."

@tool
def delete_records() -> str:
    """Delete all customer records."""
    return "All customer records have been deleted."

llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_agent(
    model=llm,
    tools=[search_web, send_email, delete_records],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records": True,
                "send_email": True,
                "search_web" : False
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
Search the web for the latest AI news.

Then send an email to amir@gmail.com with the subject "AI Update"
and the body "Here are today's AI news."

Finally, delete all customer records.
"""
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": "thread-1"
        }
    }
)

print(result)