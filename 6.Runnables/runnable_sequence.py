from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a joke about {topic}",
    input_variables=["topic"]
)
model = ChatGroq(model_name="llama-3.3-70b-versatile")

parser = StrOutputParser()


prompt2 = PromptTemplate(
    template = "Explain the joke: {joke}",
    input_variables=["joke"]
)


chain1 = RunnableSequence(prompt1, model, parser , prompt2, model, parser)

result = chain1.invoke({"topic" : "programming"})

print(result)
chain1.get_graph()
