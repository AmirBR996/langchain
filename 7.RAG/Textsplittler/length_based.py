from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
path = "tocviva.pdf"
loader = PyPDFLoader(path)
docs = loader.load()  

chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk.page_content)