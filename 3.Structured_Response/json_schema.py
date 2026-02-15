from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict , Annotated , Optional , Literal
from pydantic import BaseModel , Field

load_dotenv()

model = ChatGroq(model_name = "llama-3.3-70b-versatile")


json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}
    
prompt = """ The iPhone 12 has a new design that’s the biggest change we've seen since the iPhone X way back in 2017. Gone are the curved sides of the last few iPhones and in comes a squarer look with flat aluminium sides. Look familiar? Well it’s a throwback to the iPhone 4 - and we think it looks great.

There’s no doubt the curvy charms of the iPhone 11 and XR were aesthetically pleasing, but they were a bit slippery to hold, especially without a case. Well that’s certainly not the case with the iPhone 12. It feels solid in your hand and the metal sides definitely enable you to get a firmer grip. You might still want to get a protective cover though, just in case.
"""

structured_model = model.with_structured_output(json_schema ,strict=True)

response = structured_model.invoke(prompt)

print(response)

