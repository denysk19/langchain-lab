# IKB LangChain Lab

A minimal, production-quality RAG (Retrieval-Augmented Generation) chatbot that runs entirely in the console. Features persistent thread memory, in-memory FAISS vector store, and support for both OpenAI and vLLM providers.

## Features

- **Interactive Console Chat**: Terminal-based interface with persistent conversations
- **Thread Memory**: Multi-turn conversations with history stored in `./.state/`
- **History-Aware Retrieval**: Follow-up questions automatically rewritten using conversation context  
- **Provider Flexibility**: Switch between OpenAI and vLLM (OpenAI-compatible) endpoints
- **Document Support**: Load `.md`, `.txt`, and optionally `.pdf` files from any directory
- **In-Memory Indexing**: Fast FAISS vector store, no external database required
- **Source Citations**: Answers include bracketed references like [1], [2] to original documents

## Quick Start

### 1. Installation

Using Poetry (recommended):
```bash
cp .env_example .env
# Edit .env and set your OPENAI_API_KEY
poetry install
```

Using pip (alternative):
```bash
cp .env_example .env
# Edit .env and set your OPENAI_API_KEY
pip install langchain>=0.3.0 langchain-community>=0.3.0 langchain-openai>=0.2.0 langchain-text-splitters>=0.3.0 langgraph>=0.2.0 faiss-cpu>=1.8.0 python-dotenv>=1.0.0 pydantic-settings>=2.4.0 tqdm>=4.66.0
# Optional for PDF support:
pip install PyPDF2>=3.0.0
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

## How It Works

The system uses a **hybrid LangGraph + LangChain approach** for robust state management while keeping the workflow simple:

### **Architecture**
- **LangGraph State Machine**: Manages workflow state and provides checkpointing
- **LangChain Components**: Powers LLM interactions and document processing
- **Linear Flow**: Simple 3-step process (rewrite → retrieve → generate)

### **Workflow Steps**
1. **Document Loading**: Files from `--docs` directory are loaded and split into chunks
2. **Embedding**: Each chunk is embedded using OpenAI's text-embedding-3-small
3. **Indexing**: FAISS creates an in-memory vector index for fast similarity search
4. **Query Processing** (Enhanced LangGraph nodes):
   - **Classify Node**: Determines if query needs company docs or general knowledge
   - **Rewrite Node**: History-aware query rewriting (only if retrieval needed)
   - **Retrieve Node**: Document retrieval (only for company-specific queries)
   - **Generate Node**: Context-based OR direct answers based on query type
5. **Smart Routing**: Only retrieves documents when company-specific information is needed
6. **Citations**: Retrieval-based answers include [1], [2] references to source documents
7. **Memory**: Full conversation history maintained per thread with LangGraph checkpointing

### **Benefits of Enhanced Approach**
- ✅ **Robust State Management**: LangGraph handles complex state transitions
- ✅ **Error Recovery**: Built-in checkpointing and state persistence
- ✅ **Smart Routing**: Only retrieves when company-specific info is needed
- ✅ **Dual Mode**: Handles both document-based and general knowledge queries
- ✅ **Extensible**: Easy to add conditional logic or new nodes later
- ✅ **Production-Ready**: Better error handling than pure LCEL chains
- ✅ **Observable**: Detailed logging shows data flow between workflow nodes
- ✅ **Efficient**: Saves API calls by skipping unnecessary retrieval

## Workflow Logging

The hybrid implementation includes comprehensive logging to help you understand how data flows through the LangGraph nodes:

### **Enable Logging**
```bash
# Basic verbose logging
python scripts/chat.py --verbose

# Maximum detail logging
python scripts/chat.py --log-level DEBUG
```

### **What You'll See**
With verbose logging enabled, you'll see detailed information about:

- **State Initialization**: Input question and message history
- **Rewrite Node**: How queries are rewritten using conversation context
- **Retrieve Node**: Which documents are found and their sources
- **Generate Node**: How the final answer is constructed with citations
- **State Transfers**: Data passed between each workflow node
- **Memory Management**: How conversation history is loaded and saved

### **Example Log Output**

**Company-specific query (with retrieval):**
```
🚀 Processing: 'How many vacation days?'
🎯 Classifying query: 'How many vacation days?'
📋 Classification: RETRIEVE → needs_retrieval=True
🔄 Query: 'How many vacation days?' (no history)
🔍 Retrieving docs for: 'How many vacation days?'
📚 Found 2 docs: company-policies.md, benefits-overview.txt
🤖 Generating retrieval-based answer
💬 Generated answer (156 chars)
✅ Complete (1.45s)
```

**General knowledge query (direct answer):**
```
🚀 Processing: 'What is the capital of France?'
🎯 Classifying query: 'What is the capital of France?'
📋 Classification: DIRECT → needs_retrieval=False
🔄 Query: 'What is the capital of France?' (direct answer - no rewrite needed)
🔍 Skipping retrieval (direct answer)
🤖 Generating direct answer (general knowledge)
💬 Generated answer (89 chars)
✅ Complete (0.78s)
```

## Cost Considerations

- **Embeddings**: ~$0.0001 per 1K tokens (one-time cost during startup)
- **LLM Queries**: Varies by provider and model
  - OpenAI GPT-4o-mini: ~$0.0015 per 1K input tokens
  - vLLM: Cost depends on your hosting setup
- **Storage**: Vector index stored in memory only (no persistent storage costs)

### Cost Optimization Tips

- Use smaller embedding models: `text-embedding-3-small` vs `text-embedding-3-large`
- Use efficient chat models: `gpt-4o-mini` vs `gpt-4`
- Reduce chunk overlap and size for fewer embeddings
- Use local vLLM deployment for zero API costs

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

### Project Structure

```
ikb-langchain-lab/
├── scripts/
│   └── chat.py                 # Main interactive console app
├── src/adapters/
│   └── vllm_client_adapter.py  # Optional direct vLLM client
├── docs/                       # Sample documents  
├── tests/
│   └── test_rag_smoke.py       # Basic functionality test
├── .state/                     # Thread memory storage (created at runtime)
├── pyproject.toml             # Poetry dependencies
├── .env_example               # Environment template
└── README.md                  # This file
```

## License

MIT License - see LICENSE file for details.

