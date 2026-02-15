from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chatmodel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("Blog post generator")
print("Provide ideas or topics for blog post. Type exit to finish")

topic = input("Enter blog post topic: ")

# Static system + human prompt template
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a professional blog writer. Help generate informative, engaging and well-structured blog post about a {topic}"
    ),
    HumanMessagePromptTemplate.from_template(
        "Write a detailed blog post about {topic}"
    )
])

chat_history = []

while True:
    user_input = input("\nIdeas or instruction (or type exit): ")
    if user_input.lower() == "exit":
        print("Exiting blog post generator.")
        break

    # Base messages from template
    messages = chat_prompt_template.format_messages(topic=topic)

    # Add previous chat history
    messages.extend(chat_history)

    # Add current user instruction as literal HumanMessage

    # Generate AI response
    response = chatmodel.invoke(messages)

    print("\nBlog Post Content:\n", response.content)

    # Save conversation to history for context
    chat_history.append(response)
