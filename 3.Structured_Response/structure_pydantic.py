from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict , Annotated , Optional , Literal
from pydantic import BaseModel , Field

load_dotenv()

model = ChatGroq(model_name = "llama-3.3-70b-versatile")

class Schema(BaseModel):
    key_themes: list[str] = Field(..., description="Key themes from the review")
    summary: str = Field(..., description="Brief summary of the review")
    sentiment: str = Field(..., description="Sentiment: positive or negative")
    pros: Optional[list[str]] = Field(default=None, description="List of pros")
    cons: Optional[list[str]] = Field(default=None, description="List of cons")

    
prompt = """ The iPhone 12 has a new design that’s the biggest change we've seen since the iPhone X way back in 2017. Gone are the curved sides of the last few iPhones and in comes a squarer look with flat aluminium sides. Look familiar? Well it’s a throwback to the iPhone 4 - and we think it looks great.

There’s no doubt the curvy charms of the iPhone 11 and XR were aesthetically pleasing, but they were a bit slippery to hold, especially without a case. Well that’s certainly not the case with the iPhone 12. It feels solid in your hand and the metal sides definitely enable you to get a firmer grip. You might still want to get a protective cover though, just in case.
"""

structured_model = model.with_structured_output(Schema ,strict=True)

response = structured_model.invoke(prompt)

print(response)

