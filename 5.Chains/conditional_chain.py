from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# model = ChatGroq(model_name="llama-3.1-8b-instant")  # use 8B for speed

model = ChatGroq(model_name="llama-3.3-70b-versatile")


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback text into positive or negative.

{format_instructions}

Feedback:
{feedback}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    },
)

classifier_chain = prompt1 | model | parser2   # ✅ use Pydantic parser

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment"),
)

final_chain = classifier_chain | branch_chain

output = final_chain.invoke({
    "feedback": "Every thing in the phone is best i will give 4.5 out of 5"
})

print(output)

final_chain.get_graph().print_ascii()