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

# Chunking Strategy (optional)
CHUNKING_METHOD=token            # token, char, or semantic
# CHUNKING_METHOD=semantic       # LLM-powered section-aware chunking
# SEMANTIC_CHUNKING_LLM_MODEL=gpt-4o-mini
# SEMANTIC_CHUNKING_ENABLE_LLM=true
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
├── chat.py                 # Main chatbot application
├── evaluate.py             # RAG evaluation framework
└── test_retrieval.py       # Interactive retrieval debugging tool

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
3. **Chunked with configurable strategy:**
   - **Token-based** (default): Fast, token-aware splitting
   - **Character-based**: Simple character splitting
   - **Semantic** (🆕): LLM-powered section-aware chunking
   - **Sentence-window** (🆕): Precise small chunks with expandable context
4. Embedded using OpenAI or local embeddings
5. Stored in MemoryStore with multi-tenancy context
6. Retrieved via LangChain adapter

## Advanced Features

### 🆕 Semantic Chunking V2

**LLM-powered intelligent chunking** that **guarantees single-topic chunks** - never mixes sections!

#### Why Use Semantic Chunking?

**Problem with traditional chunking:**
```
❌ Chunk mixing sections: "...overtime payments. 5: Leave Policy You are entitled..."
❌ Chunks split mid-concept: "...26 days of leave [chunk ends]"
❌ Lost context: Which section does this chunk belong to?
```

**Semantic chunking solution:**
```
✅ Section-aware: Never mixes "Overtime" with "Leave Policy"
✅ Complete concepts: Keeps related information together
✅ Rich metadata: Each chunk knows its section and hierarchy
✅ Universal: Works with any structured document
```

#### Quick Start

```bash
# Enable in .env
CHUNKING_METHOD=semantic
SEMANTIC_CHUNKING_LLM_MODEL=gpt-4o-mini
SEMANTIC_CHUNKING_ENABLE_LLM=true  # or false for regex fallback

# Use normally - chunking happens automatically
poetry run python scripts/chat.py --docs docs/
poetry run python scripts/evaluate.py --dataset golden_dataset/ --docs docs/
```

#### How It Works

1. **LLM analyzes document** (one-time, ~$0.01 per doc)
2. **Detects sections** (Schedule 1:, Part A:, Section 1., etc.)
3. **Creates intelligent chunks** that respect boundaries
4. **Adds section context** to each chunk

#### Cost & Performance

- **LLM mode**: ~$0.01 per document, 2-5 seconds (best quality)
- **Regex mode** (`ENABLE_LLM=false`): Free, instant (good quality)
- **One-time cost**: Structure detected once at ingestion

#### Testing Semantic Chunks

```bash
# Save chunks to see the difference
poetry run python scripts/test_retrieval.py \
  --docs docs/ \
  --save-chunks semantic_chunks.txt

# Each chunk will show:
# - Complete sections (never mixed)
# - Section headers for context
# - Metadata: section_title, level, is_complete
```

#### Examples

See `src/ingestion/examples/semantic_chunking_example.py` for:
- Basic usage
- LLM vs regex modes
- Comparison with token chunking
- PDF ingestion

Full documentation: 
- `src/ingestion/SEMANTIC_CHUNKING.md` - Complete guide
- `SEMANTIC_CHUNKING_V2_IMPROVEMENTS.md` - V2 improvements (fixed multi-topic chunks!)

### 🆕 Sentence-Window Chunking

**Precision retrieval with expandable context** - best for factual Q&A and precise queries.

#### How It Works

```
1. Create SMALL chunks (50 tokens) for precise matching
2. Store surrounding context (±2 sentences) in metadata
3. At retrieval: Return expanded chunk with context automatically
```

#### Why Use Sentence-Window?

**Best for:**
- ✅ Factual questions ("What is X?", "How much is Y?")
- ✅ Precise information extraction
- ✅ Short, specific answers
- ✅ Q&A systems

**Example:**
```
Small chunk (50 tokens): "Overtime is paid at 1.5x normal rate."
Expanded (retrieval): "Base salary paid on 25th. Overtime is paid at 1.5x 
                       normal rate. You must get approval before overtime."
```

#### Quick Start

```bash
# Enable in .env
CHUNKING_METHOD=sentence-window
SENTENCE_WINDOW_CHUNK_TOKENS=50
SENTENCE_WINDOW_SENTENCES=2

# Use normally - expansion happens automatically
poetry run python scripts/chat.py --docs docs/
```

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SENTENCE_WINDOW_CHUNK_TOKENS` | 50 | Size of core chunks (smaller = more precise) |
| `SENTENCE_WINDOW_SENTENCES` | 2 | Context sentences before/after (0-5 recommended) |

#### When to Use Each Method

| Method | Best For | Chunk Size | Context |
|--------|----------|------------|---------|
| **Token** | General purpose | Fixed 450 | Good |
| **Semantic** | Structured documents | Variable (section-based) | Excellent |
| **Sentence-window** | Factual Q&A | Small 50 + window | Precise + Context |

## Debugging Tools

### Retrieval Testing Tool

Interactive tool for testing and debugging document retrieval. Useful for understanding what chunks are retrieved, their similarity scores, and whether the right documents are being found.

#### Features

- **Interactive Mode**: Test multiple queries without reloading documents
- **Distance Scores**: See exact L2 distance scores for each chunk (lower = more relevant)
- **Full Metadata**: View source file, page number, chunk index, document ID
- **Content Preview**: See complete text of retrieved chunks
- **Adjustable Top-K**: Change the number of results on the fly
- **Single Query Mode**: Test specific queries non-interactively

#### Usage

**Interactive Mode (Recommended):**

```bash
# Start interactive testing session
poetry run python scripts/test_retrieval.py --docs docs/ --k 5

# In interactive mode, you can:
query (k=5)> How many days of annual leave do I get?
query (k=5)> /k 3          # Change to top-3 results
query (k=3)> What is the refund policy?
query (k=3)> /exit         # Exit
```

**Single Query Mode:**

```bash
# Test a single query and exit
poetry run python scripts/test_retrieval.py \
  --docs docs/ \
  --k 5 \
  --query "How many days of annual leave do I get?"
```

**With Custom Chunking:**

```bash
# Test with different chunk sizes
poetry run python scripts/test_retrieval.py \
  --docs docs/ \
  --k 5 \
  --chunk-size 600 \
  --chunk-overlap 100
```

**Save All Chunks to File:**

```bash
# Save all generated chunks to a text file for analysis
poetry run python scripts/test_retrieval.py \
  --docs docs/ \
  --save-chunks chunks_analysis.txt

# Combine with other options
poetry run python scripts/test_retrieval.py \
  --docs docs/ \
  --chunk-size 400 \
  --chunk-overlap 80 \
  --save-chunks chunks_400_80.txt
```

This creates a detailed text file showing every chunk with:
- Chunk number and ID
- Document ID and source file
- Page number and chunk index
- Full content of each chunk
- Character count per chunk

#### Sample Output

```
🔍 QUERY: How many days of annual leave do I get?
================================================================================

📊 Retrieved 5 chunks:

────────────────────────────────────────────────────────────────────────────────
🔢 RANK #1
────────────────────────────────────────────────────────────────────────────────
📄 Source: bank_handbook.pdf
🆔 Document ID: bank_handbook
📑 Chunk ID: bank_handbook::5
📊 Distance Score: 0.8542 (lower is better)
📖 Page: 12 | Chunk Index: 5
────────────────────────────────────────────────────────────────────────────────
📝 CONTENT:
────────────────────────────────────────────────────────────────────────────────
  Schedule 1: Core and Other Leave
  
  You are entitled to 26 working days' core leave in each Benefits Year
  (along with public holidays recognised by the Bank), or pro rata if your
  employment begins or ends part way through a Benefits Year...
```

#### Debugging Workflow

1. **Start Interactive Mode**
   ```bash
   poetry run python scripts/test_retrieval.py --docs docs/
   ```

2. **Test Your Queries**
   - Try queries from your golden dataset
   - Test edge cases and variations
   - Compare different phrasings

3. **Analyze Chunking Quality**
   - Save chunks to file: `--save-chunks chunks.txt`
   - Review how documents are being split
   - Check if chunks are breaking at good boundaries
   - Verify chunks don't mix different sections

4. **Analyze Retrieval Results**
   - Are the right chunks ranking highest?
   - What are the distance scores?
   - Is chunking preserving important context?

5. **Experiment with Parameters**
   - Try different `--chunk-size` values
   - Adjust `--chunk-overlap`
   - Change `--k` to see score distribution
   - Save chunks for each configuration to compare

6. **Compare with Evaluation Results**
   - Use findings to tune your RAG pipeline
   - Identify documents that need better preprocessing
   - Optimize embedding and chunking strategies

7. **Integrate with Evaluation**
   - Test queries from `golden_dataset/` to understand why they pass/fail
   - Use same chunking parameters as your evaluation runs
   - Compare retrieval results with `debug_retrieved_context` from `evaluate.py`

#### Common Issues to Debug

**High Distance Scores (>1.5)**
- Embeddings may not capture the semantic meaning well
- Consider different embedding models
- Check if preprocessing is removing important context
- Note: Lower distance = better match (L2 distance)

**Wrong Chunks Retrieved**
- Chunk size may be too small/large
- Try adjusting overlap
- Review how documents are being split

**Missing Expected Documents**
- Check if documents were ingested correctly
- Verify tenant_id and visibility settings
- Test with simpler queries first

#### Command-Line Options

```bash
--docs DIRECTORY           # Documents directory (default: "docs")
--k NUMBER                 # Number of chunks to retrieve (default: 5)
--query TEXT               # Single query for non-interactive mode
--save-chunks FILEPATH     # Save all generated chunks to a text file
--chunk-size NUMBER        # Override chunk size
--chunk-overlap NUMBER     # Override chunk overlap
--tenant-id ID             # Tenant ID (default: "default-tenant")
--owner-user-id ID         # Owner ID (default: "admin")
--visibility SCOPE         # "org" or "private" (default: "org")
```

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
│   ├── chat.py            # Main chatbot application
│   ├── evaluate.py        # RAG evaluation framework
│   └── test_retrieval.py  # Interactive retrieval testing
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

#### Debug Mode

Enable detailed debug logging to capture intermediate workflow steps and timing information:

```bash
# Run evaluation with debug information
poetry run python scripts/evaluate.py \
  --dataset golden_dataset/bank_golden_dataset \
  --docs docs/ \
  --log-debug-information
```

When enabled, the CSV output includes 15 additional debug columns:

| Debug Column | Description |
|-------------|-------------|
| `debug_classification_latency` | Time spent classifying if retrieval is needed (seconds) |
| `debug_rewrite_latency` | Time spent rewriting the query (seconds) |
| `debug_retrieval_latency` | Time spent retrieving documents (seconds) |
| `debug_generation_latency` | Time spent generating the answer (seconds) |
| `debug_retrieved_sources` | Source filenames of retrieved documents |
| `debug_retrieved_doc_ids` | Document IDs of retrieved chunks |
| `debug_chunk_scores` | Similarity scores of retrieved chunks |
| `debug_original_query_length` | Character length of original query |
| `debug_rewritten_query_length` | Character length of rewritten query |
| `debug_context_length` | Character length of retrieved context |
| `debug_answer_length` | Character length of generated answer |
| `debug_workflow_state` | Step-by-step workflow execution trace |
| `debug_classification_prompt` | Full prompt sent to LLM for classification |
| `debug_rewrite_prompt` | Full prompt sent to LLM for query rewriting |
| `debug_generation_prompt` | Full prompt sent to LLM for answer generation |

**Use Cases:**
- **Performance Optimization**: Identify bottlenecks in the RAG pipeline
- **Quality Analysis**: Understand how queries transform through the workflow
- **Debugging**: Troubleshoot issues in specific workflow stages
- **Research**: Analyze relationship between query/context/answer characteristics
- **Prompt Engineering**: Review actual prompts sent to LLM for optimization

**Example Analysis:**

```python
import pandas as pd

# Load debug results
df = pd.read_csv('experiments/20251019_120000__debug_run/results.csv')

# Identify performance bottlenecks
print("Average latency by stage:")
print(f"  Classification: {df['debug_classification_latency'].mean():.3f}s")
print(f"  Rewrite: {df['debug_rewrite_latency'].mean():.3f}s")
print(f"  Retrieval: {df['debug_retrieval_latency'].mean():.3f}s")
print(f"  Generation: {df['debug_generation_latency'].mean():.3f}s")

# Analyze query rewriting impact
df['query_length_change'] = df['debug_rewritten_query_length'] - df['debug_original_query_length']
print(f"\nAverage query length change: {df['query_length_change'].mean():.1f} chars")

# Find slowest queries
slowest = df.nlargest(5, 'debug_generation_latency')[['question', 'debug_generation_latency', 'debug_context_length']]
print("\nSlowest generations:")
print(slowest)

# Analyze prompts for a specific question
question_idx = 0  # First question
print(f"\n\nPrompts for question: {df.iloc[question_idx]['question']}")
print("\n=== Classification Prompt ===")
print(df.iloc[question_idx]['debug_classification_prompt'])
print("\n=== Generation Prompt ===")
print(df.iloc[question_idx]['debug_generation_prompt'][:500] + "...")  # First 500 chars
```

**Note:** Debug mode adds minimal overhead (~5-10ms per question) but provides detailed insights for troubleshooting and optimization.

### Results Format

#### results.csv

CSV file with the following standard columns (additional debug columns available with `--log-debug-information`):

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
8. **Use debug mode** (`--log-debug-information`) when troubleshooting or optimizing performance

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