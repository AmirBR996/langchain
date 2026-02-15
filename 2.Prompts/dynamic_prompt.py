from langchain_core.prompts import PromptTemplate  

static_prompt = PromptTemplate(
    input_Variables = ["topic" , "style"],
    template = "Write a short fun fact about {topic} in a {style} style"
)

prompt_text = static_prompt.format(topic = "nepal" , style = "roast")

print(prompt_text)