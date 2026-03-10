from langchain_core.documents import Document
from langchain_community.vectorstores import chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

document = [
    Document(
        page_content="Kathmandu is the capital city of Nepal. It is known for its historic temples, vibrant culture, and UNESCO World Heritage Sites like Swayambhunath and Pashupatinath Temple.",
        metadata={"category": "City"}
    ),
    Document(
        page_content="Mount Everest, located in Nepal, is the highest mountain in the world with a height of 8848.86 meters. It attracts climbers and adventurers from around the globe.",
        metadata={"category": "Mountain"}
    ),
    Document(
        page_content="Lumbini is the birthplace of Lord Buddha and one of the most important pilgrimage sites in the world. It is located in the Rupandehi district of Nepal.",
        metadata={"category": "Religious Site"}
    ),
    Document(
        page_content="Pokhara is a beautiful tourist city in Nepal known for its lakes, scenic mountain views, and adventure sports like paragliding and trekking.",
        metadata={"category": "Tourism"}
    ),
    Document(
        page_content="Chitwan National Park is a famous wildlife reserve in Nepal. It is home to endangered animals such as the one-horned rhinoceros and the Bengal tiger.",
        metadata={"category": "National Park"}
    )
]


embedding = HuggingFaceEmbeddings()

vector_store = chroma.from_documents(document, embedding , collection_name='nepal_collection')

retriever = vector_store.as_retriever(vectorstore=vector_store , search_kwargs={"k": 2})

query = "Which place in Nepal is famous for wildlife?"

results = retriever.invoke(query)

for i , doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()


