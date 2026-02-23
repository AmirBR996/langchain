from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence , RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a joke about {topic}",
    input_variables=["topic"]
)
model = ChatGroq(model_name="llama-3.3-70b-versatile")
parser = StrOutputParser()

passthrough = RunnablePassthrough()

prompt2 = PromptTemplate(
    template = "Explain the joke: {joke}",
    input_variables=["joke"]
)

joke_generation_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' :RunnablePassthrough ,
    'explanation' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generation_chain, parallel_chain)

output = final_chain.invoke({"topic" : "cricket"})

# print("Generated Joke:")
# print(output['joke'])
# print("-----------------------------")
# print("Explanation:")
# print(output['explanation'])
# final_chain.get_graph() 

print(output)