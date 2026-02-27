from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)

load_dotenv()

model = ChatGroq(model_name="llama-3.3-70b-versatile")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Tell me a joke about {topic}",
    input_variables=["topic"]
)

def word_counter(text):
    return len(text.split())

runnable_word_count = RunnableLambda(word_counter)

# First: Generate joke
joke_chain = RunnableSequence(
    prompt,
    model,
    parser
)

# Then: Run joke + word count in parallel
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": runnable_word_count
})

# Final chain
final_chain = RunnableSequence(
    joke_chain,
    parallel_chain
)

output = final_chain.invoke({
    "topic": "programming"
})

print(output)