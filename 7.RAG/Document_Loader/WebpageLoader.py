from langchain_community.document_loaders import WebBaseLoader

url = "quotes.toscrape.com"

loader = WebBaseLoader(
    web_paths=[url],
)

docs = loader.load()
print(docs[0].page_content[:500])