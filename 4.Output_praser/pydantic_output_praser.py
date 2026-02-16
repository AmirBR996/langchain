from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Initialize model
model = ChatGroq(model_name="llama-3.3-70b-versatile")

# Define schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

# Create prompt template
template = PromptTemplate(
    template="Generate the name, age, and city of a fictional {place} person.\n{format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = template.format(place="nepali")  # safer than .invoke()

# Call model
result = model.invoke(prompt)

# Parse structured output
final_result = parser.parse(result.content.strip())

print(final_result)
