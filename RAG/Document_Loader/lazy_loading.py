from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader = DirectoryLoader(
    path = "",
    glob = "**/*.pdf",
    loader_cls= PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

# if we can many documnet to load then we can use lazy loader to load the document one by one and we can use the document metadata to filter the document that we want to load.
