from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model = "Llama-3.1-8b-instant")

response = llm.invoke("What is the capitalof nepal?")

print(response.content)