from langchain_community.retrievers import WikipediaRetriever

# Create retriever
retriever = WikipediaRetriever(top_k_results=3, lang="ne")

# Query in Nepali
query = "नेपाल"

# Retrieve documents
docs = retriever.invoke(query)

print(len(docs), "documents retrieved.\n")

# Print results
for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content[:500]}...\n")