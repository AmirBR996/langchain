from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict , Annotated , Optional

load_dotenv()

model = ChatGroq(model_name = "llama-3.3-70b-versatile")


class Schema(TypedDict):
    key_themes : Annotated[list[str] , "must write down all the key themes discussed in the review"]
    summary : Annotated[str , "must write down the brief summary of the review"]
    sentiment : Annotated[str , "must return sentiment either positive or negative"]
    pros : Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons : Annotated[Optional[list[str]],"Write down all the cons inside a list"]
    
prompt = """ The iPhone 12 has a new design that’s the biggest change we've seen since the iPhone X way back in 2017. Gone are the curved sides of the last few iPhones and in comes a squarer look with flat aluminium sides. Look familiar? Well it’s a throwback to the iPhone 4 - and we think it looks great.

There’s no doubt the curvy charms of the iPhone 11 and XR were aesthetically pleasing, but they were a bit slippery to hold, especially without a case. Well that’s certainly not the case with the iPhone 12. It feels solid in your hand and the metal sides definitely enable you to get a firmer grip. You might still want to get a protective cover though, just in case.
"""

structured_model = model.with_structured_output(Schema)

response = structured_model.invoke(prompt)

print(response)

