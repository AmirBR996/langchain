from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch
)

load_dotenv()

model = ChatGroq(model_name="llama-3.3-70b-versatile")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a detail report about {topic}",
    input_variables=["topic"]
)   

prompt2 = PromptTemplate(
    template = "Write a summary about {topic}",
    input_variables=["topic"]
)

# LECL : langchain expression language
report_generation_chain = prompt1 | model | parser

branching_chain = RunnableBranch(
    (lambda x : len(x.split()) > 100 , RunnableSequence(prompt2 , model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_generation_chain,
    branching_chain
)

output = final_chain.invoke({"topic" : "The impact of AI on modern society"})
print(output)
