# Local RAG with Ollama + ChromaDB

This is a **naive but functional Retrieval-Augmented Generation (RAG)** pipeline that runs **fully locally**

It combines:
- **ChromaDB** for vector storage and retrieval
- **LangChain text splitting** for chunking documents
- **Ollama** for local LLM inference and embeddings (`nomic-embed-text` and `llama3.1`)

## What It Does
- Loads plain text documents from an `articles/` directory
- Splits them into manageable chunks
- Embeds each chunk using `nomic-embed-text`
- Stores chunks in a persistent ChromaDB collection `chroma_persistent`
- On user query:
  - Retrieves top 5 semantically relevant chunks
  - Feeds them as system context into `llama3.1`
  - Generates a context-aware response

## Why Local + Naive?
- Avoids hallucinations by grounding answers in retrieved documents
- No reliance on cloud services or OpenAI APIs

## Example Use Case
This can serve as a starting point for:
- Personal knowledge bases
- Offline document Q&A
- Experiments with local RAG architectures

## Tech Stack
- Python
- Ollama (`llama3.1`, `nomic-embed-text`)
- ChromaDB (persistent client)
- LangChain (text splitter)
- uuid

## Setup
```bash
# 1. Install Ollama and run the models locally
ollama run llama3
ollama pull nomic-embed-text

# 2. Clone this repo
git clone https://github.com/prasbb/local-naive-rag-pipeline-ollama-chromadb

# 3. Set up Python environment
pip install -r requirements.txt

# 4. Add text files to the `articles/` folder (or replace entirely )

# 5. Run the script
python app.py
