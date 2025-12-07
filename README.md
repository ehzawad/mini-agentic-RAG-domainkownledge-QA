# Mini Agentic RAG: Domain Knowledge Q&A System

A mini agentic Retrieval-Augmented Generation (RAG) system designed for domain-specific Q&A, specifically focused on NIST security publications. Built with GPT-4o and FAISS vector search.

## Features

- **Document Processing**: Extracts and chunks PDF documents for efficient retrieval
- **Vector Search**: FAISS-based similarity search for relevant context retrieval
- **Agentic Orchestration**: Self-reflective agent with critic loop for accuracy
- **Azure OpenAI Integration**: Leverages GPT-4o and text-embedding-ada-002
- **Interactive Chat**: Command-line interface for continuous Q&A sessions
- **Secure Configuration**: Environment variable-based API key management

## Architecture

- `document_processor.py` - PDF extraction and text chunking
- `vector_store.py` - FAISS index creation and retrieval
- `agentic_rag.py` - Agent orchestrator with critic loop
- `minirag.py` - Azure OpenAI API wrapper functions
- `chat.py` - Interactive command-line chat interface
- `chunks.json` - Processed text chunks
- `faiss_index.bin` - Vector index
- `chunk_metadata.json` - Chunk metadata mapping

## Setup

### Prerequisites

- Python 3.8+
- Azure OpenAI API access

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ehzawad/mini-agentic-RAG-domainkownledge-QA.git
cd mini-agentic-RAG-domainkownledge-QA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

Required environment variables in `.env`:
```
AZURE_OPENAI_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
CHAT_DEPLOYMENT=gpt-4o
EMBED_DEPLOYMENT=text-embedding-ada-002
CHAT_API_VERSION=2025-01-01-preview
EMBED_API_VERSION=2023-05-15
```

## Usage

### Step 1: Process PDFs
Process PDF documents and create text chunks (if `chunks.json` doesn't exist):
```bash
python document_processor.py
```

### Step 2: Build FAISS Index
Create the vector index for similarity search:
```bash
python vector_store.py --build
```

### Step 3: Query the System
Ask questions directly:
```bash
python agentic_rag.py "What is zero trust architecture?"
```

### Step 4: Interactive Chat
Start a command-line chat session:
```bash
python chat.py
```

## How It Works

1. **Query Processing**: User submits a question
2. **Tool Calling**: GPT-4o decides to call `search_nist_knowledge` tool
3. **Embedding**: Query is converted to vector (1 API call)
4. **FAISS Search**: Retrieves top-k relevant chunks
5. **Context Augmentation**: Relevant chunks are passed to GPT-4o
6. **Response Generation**: GPT-4o generates answer with retrieved context
7. **Self-Reflection**: Critic loop validates accuracy before final response

## Example Queries

- "What is zero trust architecture?"
- "Explain NIST cybersecurity framework"
- "What are the key principles of identity management?"

## License

MIT License

## Author

**ehzawad**
