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

## Evaluation & Experiments

The `experiments/` directory stores evaluation experiment results. Each experiment creates a timestamped subfolder containing comprehensive results and metadata.

### Directory Structure

```
experiments/
├── README.md
├── .gitkeep
├── 20251013_101631__test_run/
│   ├── results.csv      # Detailed results for each question
│   └── metadata.txt     # Experiment configuration and statistics
└── 20251013_143000__gpt-4o-mini/
    ├── results.csv
    └── metadata.txt
```

### Running Evaluations

#### Basic Usage

```bash
# Evaluate on full golden dataset
poetry run python scripts/evaluate.py \
  --dataset golden_dataset/bank_golden_dataset \
  --docs docs/

# Evaluate on single file
poetry run python scripts/evaluate.py \
  --dataset golden_dataset/bank_golden_dataset/part-001.json \
  --docs docs/
```

#### Advanced Options

```bash
# Custom experiment name
poetry run python scripts/evaluate.py \
  --dataset golden_dataset/bank_golden_dataset \
  --experiment-name "baseline_v1" \
  --k 5

# Override chunk settings
poetry run python scripts/evaluate.py \
  --dataset golden_dataset/bank_golden_dataset \
  --chunk-size 600 \
  --chunk-overlap 120

# Full configuration
poetry run python scripts/evaluate.py \
  --dataset golden_dataset/bank_golden_dataset \
  --docs docs/ \
  --experiment-name "experiment_openai_embeddings" \
  --k 8 \
  --chunk-size 450 \
  --chunk-overlap 90 \
  --tenant-id "acme-corp" \
  --visibility org
```

### Results Format

#### results.csv

CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `question` | Original question from golden dataset |
| `llm_answer` | Answer generated by RAG system |
| `gold_answer` | Expected answer from golden dataset |
| `context` | Ground truth context from golden dataset (reference) |
| `bucket` | Category (direct, logical, process, etc.) |
| `difficulty` | Difficulty level (easy, medium, hard) |
| `timestamp` | ISO timestamp when question was processed |
| `latency_seconds` | Time taken to generate answer |
| `retrieved_context` | Context actually retrieved by RAG (truncated if >1000 chars) |
| `num_chunks_retrieved` | Count of chunks retrieved |
| `needs_retrieval` | Whether RAG retrieval was triggered |
| `rewritten_query` | Standalone query after rewriting |
| `embedding_model` | Model used for embeddings |
| `llm_model` | Model used for generation |
| `token_count_estimate` | Estimated tokens (context + question + answer) |
| `cost_estimate` | Estimated cost in USD |

#### metadata.txt

Human-readable text file containing:

- **Experiment Configuration**: Name, timestamps, duration
- **Dataset**: Path, number of questions, documents path
- **Embedding Configuration**: Provider, model, chunking settings
- **LLM Configuration**: Provider, model, temperature
- **Retrieval Configuration**: Top K, tenant ID, visibility
- **Git Versions**: Commit hashes for main repo and submodules
- **Summary Statistics**: Total questions, success rate, latency, cost

### Evaluation Configuration

Evaluation settings are controlled by:

1. **Command-line arguments**: Override specific settings per run
2. **.env file**: Default LLM provider, model, API keys
3. **Ingestion config**: Default embedding and chunking settings

#### Required Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=openai          # or 'vllm'
MODEL=gpt-4o-mini

# Embedding Configuration
EMBEDDING_PROVIDER=openai    # or 'local'
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Chunking Configuration
CHUNKING_METHOD=token
CHUNK_SIZE=450
CHUNK_OVERLAP=90

# API Keys
OPENAI_API_KEY=sk-your-key-here
```

### Analyzing Results

#### Using Python/Pandas

```python
import pandas as pd

# Load results
df = pd.read_csv('experiments/20251013_101631__test_run/results.csv')

# Basic statistics
print(f"Average latency: {df['latency_seconds'].mean():.2f}s")
print(f"Total cost: ${df['cost_estimate'].sum():.4f}")

# Group by difficulty
difficulty_stats = df.groupby('difficulty').agg({
    'latency_seconds': 'mean',
    'cost_estimate': 'sum',
    'question': 'count'
})
print(difficulty_stats)

# Find questions that needed retrieval
retrieval_questions = df[df['needs_retrieval'] == True]
print(f"Questions requiring retrieval: {len(retrieval_questions)}")
```

#### Using Command-line Tools

```bash
# Count questions by bucket
cut -d',' -f4 results.csv | sort | uniq -c

# Average latency
awk -F',' 'NR>1 {sum+=$7; count++} END {print sum/count}' results.csv

# Total estimated cost
awk -F',' 'NR>1 {sum+=$15} END {print sum}' results.csv
```

### Comparing Experiments

Compare multiple experiment runs to evaluate:

- Impact of different embedding models
- Effect of chunk size on retrieval quality
- LLM model performance differences
- Cost vs. quality tradeoffs

Example comparison script:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple experiments
exp1 = pd.read_csv('experiments/20251013_143000__baseline/results.csv')
exp2 = pd.read_csv('experiments/20251013_150000__improved/results.csv')

# Compare average latencies
print(f"Baseline avg latency: {exp1['latency_seconds'].mean():.2f}s")
print(f"Improved avg latency: {exp2['latency_seconds'].mean():.2f}s")

# Compare costs
print(f"Baseline total cost: ${exp1['cost_estimate'].sum():.4f}")
print(f"Improved total cost: ${exp2['cost_estimate'].sum():.4f}")
```

### Best Practices

1. **Use descriptive experiment names** for easier tracking
2. **Run baseline experiments first** before making changes
3. **Keep metadata.txt files** for reproducibility
4. **Version control your configs** (.env changes)
5. **Monitor costs** especially when using paid APIs
6. **Test on small datasets first** before full runs
7. **Track git commits** for reproducibility

### Troubleshooting Evaluations

#### Common Issues

**Q: Evaluation is slow**
- A: Reduce `--k` parameter or use smaller dataset for testing
- A: Use local embeddings instead of API calls

**Q: High costs**
- A: Use `gpt-4o-mini` instead of `gpt-4o`
- A: Test on small subsets first
- A: Review token estimates in results

**Q: Out of memory errors**
- A: Reduce chunk size
- A: Process documents in batches
- A: Use smaller embedding models

**Q: Model not loading**
- A: Check .env configuration
- A: Verify API keys are set
- A: For local models, ensure models are cached

### Future Enhancements

The evaluation framework can be extended with:

- Automatic metric computation (BLEU, ROUGE, semantic similarity)
- Human evaluation workflows
- A/B testing framework
- Real-time monitoring dashboard
- Automated report generation
- Integration with experiment tracking tools (MLflow, W&B)

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