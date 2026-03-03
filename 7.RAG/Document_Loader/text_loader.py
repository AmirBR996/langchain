from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize model
model = ChatGroq(model_name="llama-3.3-70b-versatile")

# Load text file (make sure file name is correct)
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

# Output parser
parser = StrOutputParser()

# Prompt template
prompt = PromptTemplate(
    template="Write a summary about the following text in 5 lines:\n{text}",
    input_variables=["text"]
)

# Create chain
chain = prompt | model | parser

# Invoke chain
output = chain.invoke({"text": docs[0].page_content})

print(output)