from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# Nepal related documents
doc1 = Document(
    page_content="Kathmandu is the capital city of Nepal. It is known for its historic temples, vibrant culture, and UNESCO World Heritage Sites like Swayambhunath and Pashupatinath Temple.",
    metadata={"category": "City"}
)

doc2 = Document(
    page_content="Mount Everest, located in Nepal, is the highest mountain in the world with a height of 8848.86 meters. It attracts climbers and adventurers from around the globe.",
    metadata={"category": "Mountain"}
)

doc3 = Document(
    page_content="Lumbini is the birthplace of Lord Buddha and one of the most important pilgrimage sites in the world. It is located in the Rupandehi district of Nepal.",
    metadata={"category": "Religious Site"}
)

doc4 = Document(
    page_content="Pokhara is a beautiful tourist city in Nepal known for its lakes, scenic mountain views, and adventure sports like paragliding and trekking.",
    metadata={"category": "Tourism"}
)

doc5 = Document(
    page_content="Chitwan National Park is a famous wildlife reserve in Nepal. It is home to endangered animals such as the one-horned rhinoceros and the Bengal tiger.",
    metadata={"category": "National Park"}
)

docs = [doc1, doc2, doc3, doc4, doc5]


# Create Vector Store
vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='nepal_collection'
)


# Add documents
vector_store.add_documents(docs)


# View stored data
vector_store.get(include=['embeddings', 'documents'])


# Similarity search
vector_store.similarity_search(
    query="Which place in Nepal is famous for wildlife?",
    k=2
)


# Similarity search with score
vector_store.similarity_search_with_score(
    query="Where was Buddha born?",
    k=2
)


# Update document example
updated_doc1 = Document(
    page_content="Kathmandu, the capital of Nepal, is the cultural and political center of the country. It is home to many ancient temples, palaces, and UNESCO heritage sites including Durbar Square and Pashupatinath Temple.",
    metadata={"category": "City"}
)

# Example update (replace with actual id returned by your DB)
vector_store.update_document(
    document_id="REPLACE_WITH_DOCUMENT_ID",
    document=updated_doc1
)


# Check database
vector_store.get(include=['embeddings', 'documents', 'metadatas'])


# Delete example
vector_store.delete(ids=["REPLACE_WITH_DOCUMENT_ID"])


# Check again
vector_store.get(include=['embeddings', 'documents', 'metadatas'])