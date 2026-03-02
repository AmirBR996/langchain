from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light."""

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

chunks = splitter.split_text(text)

print(len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
