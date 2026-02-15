from langchain_core.prompts import PromptTemplate  

static_prompt = PromptTemplate(
    input_Variables = [],
    template = "Write a short fun fact about nepal"
)

prompt_text = static_prompt.format()

print(prompt_text)