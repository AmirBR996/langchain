from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2.5",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

# Prompt 1
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a five line summary on the following text./n{text}",
    input_variables=['text']
)
prompt1 = template1.invoke("Blackhole")

ouput = model.invoke(prompt1)

prompt2 = template2.invoke(ouput.content)

result = model.invoke(prompt2)

print(result.content)