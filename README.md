# IKB LangChain Lab

A console-based RAG (Retrieval-Augmented Generation) chatbot with persistent thread memory and modular RAG workflow architecture.

## Features

- **Interactive Console Chat**: Terminal-based interface with persistent conversations
- **Thread Memory**: Multi-turn conversations with history stored in `./.state/`
- **Modular RAG Workflow**: Uses git submodule for reusable RAG components
- **Provider Flexibility**: Switch between OpenAI and vLLM endpoints
- **Document Support**: Load `.md`, `.txt`, and `.pdf` files from any directory
- **Smart Query Routing**: Only retrieves documents when needed
- **Source Citations**: Answers include bracketed references like [1], [2] to original documents

## Quick Start

### 1. Installation

```bash
# Clone with submodules
git clone --recurse-submodules <repository-url>
cd ikb-langchain-lab

# Install dependencies (includes RAG workflow submodule)
cp .env_example .env
# Edit .env and set your OPENAI_API_KEY
poetry install
```

### 2. Add Sample Documents

```bash
mkdir -p docs
echo "Sample text about PTO policy. Employees get 20 vacation days per year." > docs/sample.md
```

Or use the included sample documents in the `docs/` folder.

### 3. Start Chatting

```bash
poetry run python scripts/chat.py --thread my-conversation
```

### 4. Chat Commands

- Ask questions naturally: `"What is the PTO policy?"`
- Follow-up questions work: `"How many days?"` (uses conversation history)
- `/reset` - Clear conversation history for current thread
- `/workflow` - Show the LangGraph workflow structure
- `/exit` or `/quit` - Save and exit

## Configuration

### Environment Variables

Copy `.env_example` to `.env` and configure:

```bash
# Basic Configuration
LLM_PROVIDER=openai          # or "vllm"
MODEL=gpt-4o-mini           
OPENAI_API_KEY=sk-...        # Required for OpenAI provider + embeddings
EMBED_MODEL=text-embedding-3-small

# vLLM Configuration (only needed if LLM_PROVIDER=vllm)
VLLM_BASE_URL=http://localhost:8000
VLLM_API_KEY=sk-local        # Optional
VLLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

### Provider Switching

**OpenAI (default):**
- Set `LLM_PROVIDER=openai`
- Requires `OPENAI_API_KEY`
- Uses `MODEL` (e.g., `gpt-4o-mini`, `gpt-4`)

**vLLM (OpenAI-compatible):**
- Set `LLM_PROVIDER=vllm`
- Requires `VLLM_BASE_URL` (your vLLM server endpoint)
- Optionally set `VLLM_API_KEY` (if your server requires authentication)
- Uses `VLLM_MODEL` (fallback to `MODEL` if not set)
- Embeddings still use OpenAI, so `OPENAI_API_KEY` is still required

## Command Line Options

```bash
python scripts/chat.py [options]

Options:
  --docs DIRECTORY     Documents directory to load (default: "docs")
  --thread THREAD_ID   Thread ID for persistent memory (default: "default")  
  --k NUMBER          Number of documents to retrieve (default: 4)
  --verbose, -v       Enable verbose logging to see workflow state transfers
  --log-level LEVEL   Set logging level: DEBUG, INFO, WARNING, ERROR
```

Examples:
```bash
# Use different document folder
python scripts/chat.py --docs /path/to/my/knowledge-base

# Different conversation thread
python scripts/chat.py --thread project-alpha-discussion

# Retrieve more context per query
python scripts/chat.py --k 8

# Enable verbose logging to see workflow state transfers
python scripts/chat.py --verbose

# Debug mode with maximum logging
python scripts/chat.py --log-level DEBUG
```

## Thread Memory

Conversations are automatically saved to `./.state/<thread-id>.json` and restored when you restart with the same `--thread` parameter.

- **Persistent**: Conversations survive restarts
- **Per-Thread**: Multiple independent conversation threads  
- **Automatic**: No manual save/load required
- **JSON Format**: Human-readable storage format

## Document Loading

The system automatically loads documents from the specified directory:

- **Supported**: `.md`, `.txt` files (always), `.pdf` files (if PyPDF2 installed)
- **Recursive**: Searches subdirectories automatically
- **Chunking**: Documents split into 800-character chunks with 120-character overlap
- **Source Tracking**: Each chunk retains reference to original file

### Adding PDF Support

```bash
poetry install --extras pdf
# or
pip install PyPDF2~=3.0
```

## Architecture

The system uses a **modular RAG workflow** implemented as a git submodule:

### **Components**
- **RAG Workflow Module**: Reusable LangGraph-based workflow (git submodule)
- **Main Application**: Console interface and document management
- **Smart Query Classification**: Routes queries to retrieval or direct answer
- **Thread Memory**: Persistent conversation history per thread

### **Workflow**
1. **Query Classification**: Determines if retrieval is needed
2. **Query Rewriting**: Context-aware query enhancement (if needed)
3. **Document Retrieval**: FAISS similarity search (if needed)
4. **Answer Generation**: Context-based or direct responses
5. **Citation**: Automatic source references for retrieved content

## Debugging

Enable verbose logging to see workflow details:

```bash
# Basic verbose logging
python scripts/chat.py --verbose

# Maximum detail logging
python scripts/chat.py --log-level DEBUG
```

## Development

### Working with the RAG Submodule

The RAG workflow is implemented as a git submodule for reusability:

```bash
# Update submodule to latest
git submodule update --remote src/rag-module

# Make changes to the submodule
cd src/rag-module
# ... make changes ...
git add . && git commit -m "Update RAG workflow"
git push

# Update main project to use new submodule version
cd ../..
git add src/rag-module
git commit -m "Update RAG module reference"
```

## Troubleshooting

### Common Issues

**"No documents found"**
- Ensure documents exist in the specified `--docs` directory
- Check file extensions (`.md`, `.txt`, optionally `.pdf`)
- Verify files are not empty

**"OPENAI_API_KEY is required"**
- Copy `.env_example` to `.env`
- Set valid OpenAI API key for embeddings (required even with vLLM)

**"VLLM_BASE_URL is required"**
- When using `LLM_PROVIDER=vllm`, set the vLLM server endpoint
- Ensure vLLM server is running and accessible

**"I don't know" responses**
- The AI only answers from provided documents
- Add relevant documents to your `--docs` directory
- Try rephrasing your question
- Check if information exists in source documents

### Performance Tips

- **Startup Time**: Proportional to document count (embedding generation)
- **Memory Usage**: Scales with document corpus size (FAISS index in RAM)
- **Query Speed**: Fast retrieval (FAISS) + LLM latency

## Development

### Running Tests

```bash
poetry run pytest tests/
```

### Code Quality

```bash
poetry run ruff check scripts/ src/
poetry run ruff format scripts/ src/
```

## License

MIT License - see LICENSE file for details.

