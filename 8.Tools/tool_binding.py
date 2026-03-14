from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import dotenv 

dotenv.load_dotenv()  
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b


llm = ChatGroq(model="llama-3.1-8b-instant", max_tokens= 50)


llm_with_tools = llm.bind_tools([multiply])

print("thinking ...")

query = HumanMessage(content="What is 5 multiplied by 3?")

messages = [query]

response = llm_with_tools.invoke(messages)
messages.append(response)
tool_result = multiply.invoke(response.tool_calls[0])

messages.append(tool_result)

final_result = llm_with_tools.invoke(messages)

print(final_result.content)

