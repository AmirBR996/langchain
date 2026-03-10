from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

document = [
    Document(
        page_content="Kathmandu is the capital city of Nepal. It is known for its historic temples and UNESCO sites.",
        metadata={"category": "City"}
    ),
    Document(
        page_content="Mount Everest in Nepal is the highest mountain in the world.",
        metadata={"category": "Mountain"}
    ),
    Document(
        page_content="Lumbini is the birthplace of Lord Buddha.",
        metadata={"category": "Religious Site"}
    ),
    Document(
        page_content="Pokhara is famous for lakes and adventure tourism.",
        metadata={"category": "Tourism"}
    ),
    Document(
        page_content="Chitwan National Park is home to the one-horned rhinoceros and Bengal tiger.",
        metadata={"category": "Wildlife"}
    )
]

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(document, embedding)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.9}
)

query = "Which place in Nepal is famous for wildlife?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()