from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence
load_dotenv()

llm1 = ChatGroq(model_name="llama-3.3-70b-versatile")
llm2 = ChatGroq(model_name="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic} for twitter  in 5 lines",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template = "Generate a post about {topic} for LinkedIn  in five lines",
    input_variables=["topic"]
)
parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, llm1, parser),
    'linkedin_post' : RunnableSequence(prompt2, llm2, parser)
})

output = parallel_chain.invoke({"topic" : "AI in healthcare"})
print("Tweet:")
print(output['tweet'])
print("-----------------------------")
print("LinkedIn Post:")
print(output['linkedin_post'])