# RAG Chatbot with Memory & Document Ingestion

Console-based RAG chatbot with persistent memory, modular architecture, and advanced PDF ingestion.

## Quick Start

```bash
# 1. Install dependencies
poetry install

# 2. Configure environment
cp .env_example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# 3. Add PDF documents
cp your-document.pdf docs/

# 4. Run chatbot
poetry run python scripts/chat.py --docs docs/
```

## Features

- **Persistent Conversations**: Thread-based memory stored in `.state/`
- **PDF Ingestion**: Dedicated ingestion module for document processing
- **In-Memory Vector Store**: Fast MemoryStore with multi-tenancy support
- **Smart Routing**: Only retrieves documents when needed
- **Multi-Provider**: Switch between OpenAI and vLLM
- **Source Citations**: Answers include document references

## Installation

### Prerequisites

- Python 3.11+
- Poetry (`curl -sSL https://install.python-poetry.org | python3 -`)
- OpenAI API key

### Setup

The project uses two git submodules (both in `src/`):
- `src/ingestion` - Document ingestion module
- `src/rag-module` - RAG workflow module

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd langchain-lab

# If submodules not initialized
git submodule update --init --recursive

# Install everything (including submodules)
poetry install
```

### Configuration

Create `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# LLM Provider (choose one)
LLM_PROVIDER=openai              # Use OpenAI
MODEL=gpt-4o-mini

# Or use vLLM
# LLM_PROVIDER=vllm
# VLLM_BASE_URL=http://localhost:8000
# VLLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

## Usage

### Basic Commands

```bash
# Run chatbot
poetry run python scripts/chat.py --docs docs/

# With verbose logging
poetry run python scripts/chat.py --docs docs/ --verbose

# Custom configuration
poetry run python scripts/chat.py \
  --docs docs/ \
  --k 10 \
  --chunk-chars 1200 \
  --chunk-overlap 200
```

### Using Poetry Shell

```bash
# Activate environment (no need to type 'poetry run' for each command)
poetry shell

python scripts/chat.py --docs docs/
python test_integration.py

# Exit when done
exit
```

### In-Chat Commands

- `/exit`, `/quit` - Save and exit
- `/reset` - Clear conversation history
- `/workflow` - Show RAG workflow diagram
- `/threads` - List all threads
- `/new [id]` - Create new thread
- `/info` - Show session info

## Command-Line Options

```bash
# Basic
--docs DIRECTORY        # Documents directory (default: "docs")
--thread THREAD_ID      # Thread ID for memory (default: "default")
--user USER_ID          # User ID (default: "default")
--k NUMBER              # Documents to retrieve (default: 8)

# Ingestion
--tenant-id ID          # Tenant ID (default: "default-tenant")
--owner-user-id ID      # Document owner (default: "admin")
--visibility SCOPE      # "org" or "private" (default: "org")
--chunk-chars NUMBER    # Chunk size (default: 1000)
--chunk-overlap NUMBER  # Overlap (default: 150)
--embed-model MODEL     # Embedding model (default: "text-embedding-3-small")

# Memory
--max-messages NUMBER   # Max messages per thread (default: 100)
--max-threads NUMBER    # Max threads per user (default: 10)

# Logging
--verbose, -v           # Verbose logging
--log-level LEVEL       # DEBUG, INFO, WARNING, ERROR
```

## Architecture

### Structure

```
src/
├── ingestion/              # Ingestion submodule
│   └── src/ingestion/
│       ├── memory_store.py # In-memory vector store
│       ├── models.py       # DocumentCtx
│       └── ingest.py       # PDF processing
├── rag-module/             # RAG workflow submodule
│   └── src/rag_workflow/
│       ├── graph.py        # LangGraph workflow
│       ├── nodes.py        # Workflow nodes
│       └── prompts.yaml    # Configurable prompts
└── adapters/
    └── memory_retriever_adapter.py  # LangChain adapter

scripts/
└── chat.py                 # Main application

docs/                       # Place PDF documents here
```

### Workflow

1. **Query Classification** - Determines if retrieval is needed
2. **Query Rewriting** - Context-aware enhancement
3. **Document Retrieval** - MemoryStore vector search
4. **Answer Generation** - Context-based or direct response
5. **Citation** - Automatic source references

### Ingestion Pipeline

1. PDFs loaded from `--docs` directory
2. Processed by `ingest_pdf()` (extract text, metadata)
3. Chunked with configurable size/overlap
4. Embedded using OpenAI embeddings
5. Stored in MemoryStore with multi-tenancy context
6. Retrieved via LangChain adapter

## Development

### Project Structure

```
langchain-lab/
├── .env                    # Environment config (not committed)
├── .env_example            # Environment template
├── pyproject.toml          # Poetry dependencies
├── poetry.lock             # Locked versions
├── docs/                   # PDF documents
├── scripts/
│   └── chat.py            # Main application
├── src/
│   ├── adapters/          # LangChain adapters
│   ├── ingestion/         # Submodule: document ingestion
│   └── rag-module/        # Submodule: RAG workflow
├── tests/
│   └── test_rag_smoke.py
└── test_integration.py    # Integration verification
```

### Testing

```bash
# Run integration test
poetry run python test_integration.py

# Run unit tests
poetry run pytest tests/

# Code quality
poetry run ruff check scripts/ src/
```

### Working with Submodules

```bash
# Update submodules to latest
git submodule update --remote

# Update specific submodule
cd src/ingestion
git pull origin main
cd ../..

# Changes are immediately available (develop mode)
poetry install  # Only if dependencies changed
```


## Examples

### Basic Chat Session

```bash
$ poetry run python scripts/chat.py --docs docs/

Loading 1 PDF files...
✅ Ingested bank_handbook.pdf: 102 chunks
Using provider: openai, model: gpt-4o-mini
✅ Ready! User: default, Thread: default

default> What is the refund policy?
bot> According to the handbook, customers have 30 days for full refunds. [1]
(time: 2.3s)

default> Are there exceptions?
bot> Yes, digital products are non-refundable. Defective items can be returned anytime. [1]
(time: 1.8s)
```

### Multi-Tenancy

```bash
poetry run python scripts/chat.py \
  --tenant-id acme-corp \
  --owner-user-id john-doe \
  --visibility org \
  --docs /path/to/company-docs/
```

### Development with Verbose Logging

```bash
poetry run python scripts/chat.py \
  --docs docs/ \
  --log-level DEBUG \
  --verbose
```

## License

MIT License - see LICENSE file for details.