# LangChain Essentials & Advanced LLM Patterns

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3%2B-121011?style=flat-square&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen?style=flat-square)]()

A production-grade, educational repository designed to master modern Large Language Model (LLM) orchestration using **LangChain**. This project covers everything from core provider integrations to advanced Retrieval-Augmented Generation (RAG), LangChain Expression Language (LCEL), custom tools, and human-in-the-loop (HITL) guardrails.

---

## 🚀 Overview

This repository is built as a hands-on learning lab and reference implementation for developers building generative AI applications. Key learning outcomes include:

* **Production-Ready Chains**: Building deterministic, type-safe pipelines using LCEL.
* **Structured Data Extraction**: Forcing LLM responses into verified schema objects (`Pydantic`, `TypedDict`, `JSON`).
* **End-to-End RAG Systems**: Implementing context-aware document loading, advanced text chunking, and semantic retrievers.
* **Safety & Control**: Implementing guardrails, PII redaction, and Human-In-The-Loop (HITL) approval pipelines.

---

## Features

### ✨ Core Capabilities

- **Multi-LLM Support**: Integration with Google Gemini, Groq (Llama), and other LLM providers
- **Prompt Engineering**: Static, dynamic, and chat-based prompt templates
- **Structured Responses**: Type-safe output handling using Pydantic and TypedDict
- **Multiple Output Parsers**: JSON, String, and Pydantic-based parsing
- **Chain Patterns**: Simple, sequential, conditional, and parallel chains
- **Advanced Runnables**: Lambda functions, parallel execution, branching, and passthrough
- **RAG Implementation**: Document loading, semantic text splitting, and retrieval patterns
- **Advanced Retrievers**: Contextual compression, MMR, multi-query, and Wikipedia retrieval
- **Vector Stores**: Chroma and other vector database integrations
- **Tools & Utilities**: Custom tool creation, currency conversion, and tool binding
- **Guardrails & Evaluation**: PII redaction, HITL middleware, and model-based evaluation
- **Interactive Learning**: Jupyter notebooks with practical exercises

## Project Structure

```
langchain/
├── 1.basic-llm/                    # LLM provider integrations
│   ├── gemini_llm.py              # Google Gemini integration
│   └── grok_llm.py                # Groq Llama integration
│
├── 2.Prompts/                      # Prompt management and templates
│   ├── static_prompt.py            # Fixed prompt templates
│   ├── dynamic_prompt.py           # Parameterized prompts
│   ├── chat_prompt_template.py     # Conversation-based prompts
│   └── project.py                  # Integrated prompt examples
│
├── 3.Structured_Response/          # Response structuring and validation
│   ├── structure_pydantic.py       # Pydantic BaseModel responses
│   ├── json_schema.py              # JSON schema definitions
│   ├── typedict.py                 # TypedDict implementations
│   └── annoted_typedict.py         # Annotated type definitions
│
├── 4.Output_praser/                # Output parsing strategies
│   ├── jsonoutputparser.py         # JSON output parsing
│   ├── pydantic_output_praser.py   # Pydantic-based parsing
│   ├── stringoutputpraser.py       # String parsing
│   └── stroutputpraser.py          # Alternative string parser
│
├── 5.Chains/                       # Chain composition patterns
│   ├── simple_chain.py             # Basic prompt|model|parser
│   ├── sequential_chain.py         # Sequential execution
│   ├── conditional_chain.py        # Conditional branching
│   └── parallel_chain.py           # Parallel execution
│
├── 6.Runnables/                    # LCEL (LangChain Expression Language)
│   ├── runnable_sequence.py        # Sequential composition
│   ├── runnable_parallel.py        # Parallel execution
│   ├── runnable_lambda.py          # Custom callable integration
│   ├── Runnable_branch.py          # Conditional logic
│   └── runnable_passthrough.py     # Input passthrough
│
├── 7.RAG/                          # Retrieval-Augmented Generation
│   ├── Document_Loader/            # Document ingestion
│   │   ├── csv_loader.py           # CSV file loading
│   │   ├── text_loader.py          # Plain text loading
│   │   ├── directory_loader.py     # Directory-based loading
│   │   ├── pyPDFloader.py          # PDF document loading
│   │   ├── WebpageLoader.py        # Web scraping
│   │   ├── lazy_loading.py         # Lazy loading patterns
│   │   ├── cricket.txt             # Sample dataset
│   │   └── data.csv                # Sample CSV data
│   │
│   ├── Retriever/                  # Retrieval strategies
│   │   ├── Contextual.py           # Contextual compression retriever
│   │   ├── MMR.py                  # Maximal Marginal Relevance
│   │   ├── multi-query-retriever.py # Multi-query retrieval
│   │   ├── vector_Store_retriever.py # Vector store retriever
│   │   └── wikipedia_retriever.py  # Wikipedia retriever
│   │
│   ├── Textsplittler/              # Text chunking strategies
│   │   ├── length_based.py         # Fixed-length chunking
│   │   ├── document_based.py       # Structure-preserving split
│   │   ├── semantic_based.py       # Semantic similarity splitting
│   │   ├── Text_Structure_based.py # Document structure aware
│   │   └── tocviva.pdf             # Sample PDF document
│   │
│   └── Vector_store/               # Vector database integrations
│       └── chroma.py               # Chroma vector store
│
├── 8.Tools/                        # LangChain tools and utilities
│   ├── currency_converter.py       # Currency conversion tool
│   ├── tool_binding.py             # Tool binding examples
│   └── Custom_Tools.ipynb          # Custom tools notebook
│
├── Guardrials/                     # Guardrails and evaluation
│   ├── built-in-guardrials.py      # Built-in guardrails
│   ├── custom(after).py            # Custom guardrails (after)
│   ├── custom(before).py           # Custom guardrails (before)
│   ├── deterministic.py            # Deterministic evaluation
│   ├── health-care-chatbot.py      # Healthcare chatbot example
│   ├── HITL-middleware.py          # Human-in-the-loop middleware
│   └── Model-based-approach.py     # Model-based guardrails
│
├── colab/                          # Jupyter notebooks
│   ├── langchain_chainproblem_at_begining.ipynb
│   ├── Runnables_in_langchain.ipynb
│   └── Built_in_tool_in_langchain.ipynb
│
├── myvenv/                         # Python virtual environment
├── requirement.txt                 # Project dependencies
├── Youtube_chatbot.py              # YouTube transcript chatbot
└── .env                            # Environment variables (not included)
```

## Prerequisites

- **Python**: 3.11 or higher
- **Package Manager**: pip or conda
- **API Keys**: 
  - Google Generative AI API key (for Gemini)
  - Groq API key (free tier available)
  - OpenAI API key (optional, for alternative models)

## Installation

### 1. Clone or Navigate to Project

```bash
cd /path/to/langchain
```

### 2. Create Virtual Environment

```bash
python3 -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

### 4. Install Additional Dependencies (if needed)

```bash
pip install langchain-experimental jupyter
```

### 5. Verify Installation

```bash
python -c "import langchain; print(f'LangChain version: {langchain.__version__}')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key_here

# Groq API
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API (optional)
OPENAI_API_KEY=your_openai_api_key_here
```

**Obtaining API Keys:**

- **Google API**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Groq API**: Sign up at [Groq Console](https://console.groq.com/keys) (free tier available)

### Load Environment Variables in Code

All examples use:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Quick Start

### 1. Basic LLM Query

```bash
cd 1.basic-llm
python gemini_llm.py
```

### 2. Simple Chain Example

```bash
cd 5.Chains
python simple_chain.py
```

### 3. Interactive Notebook

```bash
jupyter notebook colab/langchain_chainproblem_at_begining.ipynb
```

## Module Guide

### 1. **Basic LLM Integration** (`1.basic-llm/`)

Learn to connect and interact with different LLM providers.

**Key Concepts:**
- Provider initialization
- Direct invocation
- Response handling

**Example:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke("What is the capital of Nepal?")
print(response.content)
```

---

### 2. **Prompt Engineering** (`2.Prompts/`)

Master various prompt template patterns for different use cases.

**Pattern Types:**
- **Static Prompts**: Fixed templates with no variables
- **Dynamic Prompts**: Parameterized templates
- **Chat Prompts**: Multi-turn conversation templates with system messages

**Example:**
```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant about {subject}."),
    ("human", "Tell me something interesting about {subject}.")
])

messages = chat_prompt.format_messages(subject="quantum computing")
```

---

### 3. **Structured Responses** (`3.Structured_Response/`)

Enforce type safety and structured output from LLMs.

**Approaches:**
- **Pydantic Models**: Full validation with Python types
- **TypedDict**: Lightweight type hints
- **Annotated Types**: Enhanced type metadata

**Example:**
```python
from pydantic import BaseModel, Field

class ReviewAnalysis(BaseModel):
    sentiment: str = Field(..., description="positive or negative")
    key_themes: list[str]
    summary: str

structured_llm = model.with_structured_output(ReviewAnalysis, strict=True)
result = structured_llm.invoke(text)
```

---

### 4. **Output Parsers** (`4.Output_praser/`)

Convert unstructured LLM output into usable formats.

**Parser Types:**
- **JsonOutputParser**: Parse JSON-formatted strings
- **PydanticOutputParser**: Direct Pydantic model parsing
- **StrOutputParser**: Simple string extraction
- **CustomParsers**: Build your own parsing logic

---

### 5. **Chains** (`5.Chains/`)

Compose multiple LLM calls and processing steps.

**Chain Types:**

**Simple Chain:**
```python
chain = prompt | model | parser
result = chain.invoke({"topic": "Python"})
```

**Sequential Chain:**
```python
chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser
full_chain = chain1 | chain2
```

**Conditional Chain:**
Branching logic based on input conditions.

**Parallel Chain:**
Execute multiple independent chains simultaneously.

---

### 6. **Runnables (LCEL)** (`6.Runnables/`)

Use LangChain Expression Language for advanced compositions.

**Runnable Types:**
- **RunnableSequence**: Pipe multiple steps
- **RunnableParallel**: Execute in parallel
- **RunnableBranch**: Conditional execution
- **RunnableLambda**: Wrap custom functions
- **RunnablePassthrough**: Identity passthrough

**Example:**
```python
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence(
    prompt1, model, parser,
    prompt2, model, parser
)
result = chain.invoke({"topic": "AI"})
```

---

### 7. **RAG (Retrieval-Augmented Generation)** (`7.RAG/`)

Build systems that combine document retrieval with LLM generation.

#### **Document Loading** (`Document_Loader/`)

**Supported Formats:**
- CSV files
- Plain text
- PDF documents
- Web pages
- Directory contents (lazy-loaded)

**Example:**
```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="data.csv")
documents = loader.load()
```

#### **Text Splitting** (`Textsplittler/`)

**Splitting Strategies:**

1. **Length-Based**: Fixed chunk size with overlap
2. **Document-Based**: Respect document structures
3. **Semantic-Based**: Split by semantic similarity
4. **Structure-Based**: Preserve document hierarchy

**Example:**
```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(embeddings)
chunks = splitter.split_text(long_text)
```

---

### 8. **Tools** (`8.Tools/`)

Integrate LangChain tools for external API calls and utility functions.

**Capabilities:**
- Custom tool definitions using `@tool` decorator
- Tool binding and invocation patterns
- Real-world utility examples (currency conversion)

**Example:**
```python
from langchain_core.tools import tool
from langchain_groq import ChatGroq

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools([multiply])
```

---

### 9. **RAG Retrievers** (`7.RAG/Retriever/`)

Advanced retrieval strategies for RAG systems.

**Retriever Types:**
- **Contextual Compression**: Filter relevant content using LLM
- **MMR (Maximal Marginal Relevance)**: Balance relevance and diversity
- **Multi-Query**: Generate multiple query perspectives
- **Vector Store Retriever**: Direct vector similarity search
- **Wikipedia Retriever**: Query Wikipedia knowledge base

**Example:**
```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

base_retriever = vectorstore.as_retriever()
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
docs = retriever.invoke("What is Nepal?")
```

---

### 10. **Vector Stores** (`7.RAG/Vector_store/`)

Persist and query embeddings using vector databases.

**Supported Backends:**
- **Chroma**: Open-source embedding database

**Example:**
```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()
```

---

### 11. **Guardrails / Evaluation** (`Guardrials/`)

Ensure LLM outputs are safe, accurate, and aligned with requirements.

**Guardrail Types:**
- **Built-in Guardrails**: PII redaction, topic restriction
- **Custom Guardrails**: Before/after processing hooks
- **Human-in-the-Loop (HITL)**: Human approval middleware
- **Model-based Approach**: LLM-as-judge evaluation
- **Deterministic Checks**: Rule-based validation

**Example:**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=llm,
    tools=[customer_lookup],
    middleware=[PIIMiddleware()]
)
```

---

## Usage Examples

### Example 1: Fact Generation Chain

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)
model = ChatGroq(model_name="llama-3.3-70b-versatile")

# Create chain
chain = prompt | model | StrOutputParser()

# Execute
result = chain.invoke({"topic": "Nepal"})
print(result)
```

### Example 2: Structured Review Analysis

```python
from pydantic import BaseModel, Field

class ReviewSchema(BaseModel):
    sentiment: str = Field(..., description="positive or negative")
    key_themes: list[str]
    pros: list[str]
    cons: list[str]

# Use with LLM
structured_model = model.with_structured_output(ReviewSchema, strict=True)
analysis = structured_model.invoke(review_text)

print(f"Sentiment: {analysis.sentiment}")
print(f"Themes: {analysis.key_themes}")
```

### Example 3: Multi-Step Processing

```python
from langchain_core.runnables import RunnableSequence

# Step 1: Generate joke
prompt1 = PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=["topic"]
)

# Step 2: Explain joke
prompt2 = PromptTemplate(
    template="Explain why this joke is funny: {joke}",
    input_variables=["joke"]
)

# Chain them together
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
result = chain.invoke({"topic": "programming"})
```

## Technologies Used

### Core Framework
- **LangChain** (v0.1+): LLM application framework
- **LangChain-Core**: Base abstractions
- **LangChain-Community**: Third-party integrations
- **LangChain-Experimental**: Advanced features (semantic chunking, etc.)
- **LangChain-Classic**: Classic retrievers and agents

### LLM Providers
- **Google Gemini** (via `langchain-google-genai`)
- **Groq Llama** (via `langchain-groq`)

### Embeddings & Vector Stores
- **HuggingFace Embeddings** (via `langchain-huggingface`)
- **OpenAI Embeddings** (via `langchain-openai`)
- **Chroma**: Vector database for embeddings

### Utilities
- **Pydantic**: Data validation and settings management
- **python-dotenv**: Environment variable management
- **youtube-transcript-api**: YouTube transcript fetching
- **Grandalf**: Graph visualization (optional)

### Development Tools
- **Jupyter/IPython**: Interactive notebook environment
- **Python 3.11+**: Programming language

## Running Examples

### Basic LLM Examples
```bash
# Test Google Gemini
python 1.basic-llm/gemini_llm.py

# Test Groq Llama
python 1.basic-llm/grok_llm.py
```

### Prompt Examples
```bash
python 2.Prompts/chat_prompt_template.py
python 2.Prompts/dynamic_prompt.py
```

### Chain Examples
```bash
python 5.Chains/simple_chain.py
python 5.Chains/sequential_chain.py
```

### RAG Examples
```bash
python 7.RAG/Document_Loader/csv_loader.py
python 7.RAG/Textsplittler/semantic_based.py
```

### RAG Retriever Examples
```bash
python 7.RAG/Retriever/multi-query-retriever.py
python 7.RAG/Retriever/wikipedia_retriever.py
```

### Vector Store Examples
```bash
python 7.RAG/Vector_store/chroma.py
```

### Tools Examples
```bash
python 8.Tools/currency_converter.py
python 8.Tools/tool_binding.py
```

### Guardrails Examples
```bash
python Guardrials/built-in-guardrials.py
python Guardrials/health-care-chatbot.py
```

## Contributing

This is a learning/personal project. Feel free to:
- Fork and create your own variations
- Add new examples or modules
- Improve documentation
- Fix bugs or optimize code

## Future Improvements

- [x] Vector database integration (Chroma)
- [x] Advanced RAG with similarity search
- [x] Function calling/tool use examples
- [ ] Memory/conversation management examples
- [ ] Agent framework implementation
- [ ] Streaming response handling
- [ ] Performance optimization guide
- [ ] Testing strategies and examples
- [ ] Deployment patterns (Docker, cloud platforms)
- [ ] Advanced error handling and retry logic
- [ ] Cost optimization examples
- [ ] Extended documentation with theory
- [ ] Guardrails and evaluation frameworks
- [ ] YouTube transcript RAG chatbot

## Resources

- [LangChain Official Documentation](https://python.langchain.com/)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)
- [LangChain Discord Community](https://discord.gg/cU2AdqiC7h)
- [Groq API Documentation](https://console.groq.com/docs)
- [Google Generative AI Docs](https://ai.google.dev/)

---

**Last Updated**: March 2026  
**Python Version**: 3.11+  
**Status**: Active Development
