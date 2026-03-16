from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import requests
import dotenv

dotenv.load_dotenv()

# Tool 1: Multiply
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b


# Tool 2: Get conversion factor
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch conversion rate between two currencies."""
    
    url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}"
    
    response = requests.get(url)
    data = response.json()

    return data["conversion_rate"]


# Tool 3: Convert currency
@tool
def currency_converter(amount: float, base_currency: str, target_currency: str) -> float:
    """Convert currency from base to target."""
    
    rate = get_conversion_factor.invoke({
        "base_currency": base_currency,
        "target_currency": target_currency
    })  

    return amount * rate


# LLM
llm = ChatGroq(model="llama-3.1-8b-instant", max_tokens=100)

llm_with_tools = llm.bind_tools([multiply, currency_converter])

print("thinking...")

query = HumanMessage(content="Convert 100 USD to NPR")

messages = [query]

response = llm_with_tools.invoke(messages)
messages.append(response)

# Run tool
tool_call = response.tool_calls[0]
tool_result = currency_converter.invoke(tool_call)

messages.append(tool_result)

# Final answer
final_result = llm_with_tools.invoke(messages)

print(final_result.content)