from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Mount Everest, located in Nepal, is the highest mountain in the world, attracting climbers from across the globe.",
        metadata={"category": "Geography"}
    ),
    Document(
        page_content="Bitcoin is a decentralized digital currency that enables peer-to-peer transactions over the internet without a central authority.",
        metadata={"category": "Finance"}
    ),
    Document(
        page_content="The Great Barrier Reef in Australia is the largest coral reef system, home to thousands of marine species.",
        metadata={"category": "Nature"}
    ),
    Document(
        page_content="Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.",
        metadata={"category": "Technology"}
    ),
    Document(
        page_content="Leonardo da Vinci was a Renaissance polymath known for masterpieces like the Mona Lisa and The Last Supper.",
        metadata={"category": "Art & History"}
    ),
    Document(
        page_content="Yoga is a physical, mental, and spiritual practice originating from ancient India, promoting flexibility, balance, and mindfulness.",
        metadata={"category": "Health & Fitness"}
    ),
    Document(
        page_content="The Nile River, the longest river in Africa, flows through multiple countries and supports agriculture along its banks.",
        metadata={"category": "Geography"}
    ),
    Document(
        page_content="Marie Curie was the first woman to win a Nobel Prize, famous for her research on radioactivity.",
        metadata={"category": "Science & Biography"}
    ),
    Document(
        page_content="Shakespeare's plays, including Hamlet and Macbeth, are celebrated worldwide for their storytelling and exploration of human nature.",
        metadata={"category": "Literature"}
    ),
    Document(
        page_content="Electric cars are vehicles powered by electric motors and batteries, offering a sustainable alternative to traditional gasoline vehicles.",
        metadata={"category": "Technology & Environment"}
    )
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding)

similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0),
)

query = "What are some important health tips for maintaining a healthy lifestyle?"

multi_results = multi_query_retriever.invoke(query)
similarity_results = similarity_retriever.invoke(query)

print("=== MultiQueryRetriever Results ===")
for i, doc in enumerate(multi_results):
    print(f"Result {i+1}: {doc.page_content} | Metadata: {doc.metadata}")

print("\n=== SimilarityRetriever Results ===")
for i, doc in enumerate(similarity_results):
    print(f"Result {i+1}: {doc.page_content} | Metadata: {doc.metadata}")