from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGroq(model_name = "llama-3.3-70b-versatile")

# SCHEMA

class Review(TypedDict):
    summary : str
    sentiment : str

prompt = """ Nepal, officially the Federal Democratic Republic of Nepal, is
a landlocked, multi-ethnic, and multilingual country in South Asia, bordered by China to the north and India to the south, east, and west. Situated along the Himalayas, it boasts eight of the world's ten highest peaks, including the highest point on Earth, Mount Everest, making it a premier destination for mountaineering and natural beauty. Kathmandu is the capital and largest city, serving as a hub for the country’s rich, diverse culture. Geographically divided into the Himalayan, mid-hill, and Tarai regions, Nepal is famously the birthplace of Buddha (Lumbini) and features a mix of Hindu and Buddhist traditions."""

structured_model = model.with_structured_output(Review)

response = structured_model.invoke(prompt)

print(response)