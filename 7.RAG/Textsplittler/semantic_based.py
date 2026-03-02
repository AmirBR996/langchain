import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import GroqEmbeddings

# Set your Groq API key (get one free at https://console.groq.com/keys)
os.environ["GROQ_API_KEY"] = "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"  # ← replace!

text = """In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light."""

embeddings = GroqEmbeddings() 

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # balanced for this short text
    # Try "standard_deviation" if you want potentially more/fewer breaks
)

chunks = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk.strip()}\n{'-' * 60}")