from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# ===============================
# Step 1 — Transcript Ingestion
# ===============================

video_id = "Gfr50f6ZBvo"

try:
    print("Fetching transcript...")
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
    print("Transcript fetched successfully\n")

except TranscriptsDisabled:
    print("No captions available for this video.")
    exit()

# ===============================
# Step 2 — Text Splitting
# ===============================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print(f"Chunks created: {len(chunks)}\n")

# ===============================
# Step 3 — Embeddings
# ===============================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)

# ===============================
# Step 4 — Retriever
# ===============================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ===============================
# Step 5 — Groq LLM
# ===============================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ===============================
# Step 6 — Prompt
# ===============================

prompt = PromptTemplate(
    template="""
Answer using the transcript context.
If the context does not contain the answer, you may try to give a general answer.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# ===============================
# Step 7 — Formatting Function
# ===============================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ===============================
# Step 8 — RAG Chain
# ===============================

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# ===============================
# Step 9 — Chat Loop
# ===============================

print("🎥 YouTube RAG Chatbot Ready")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye 👋")
        break

    answer = main_chain.invoke(question)

    print("\nBot:", answer)
    print()