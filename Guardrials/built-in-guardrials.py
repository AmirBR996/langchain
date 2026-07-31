from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pprint import pprint

load_dotenv()

@tool
def customer_lookup(text: str) -> str:
    """Look up customer information."""
    print(f"Tool received: {text}")
    return f"Customer record found for: {text}"

llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_agent(
    model=llm,
    tools=[customer_lookup],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
Use the customer_lookup tool.

Customer Name: Amir Bhattarai
Email: amir.bhattarai@gmail.com
Phone: +9779812345678
API Key: sk-123456789abcdef

Find my customer record.
"""
            }
        ]
    }
)

pprint(result)
print(result["messages"][-1].content)