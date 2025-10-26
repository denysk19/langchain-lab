# Sentence-Window Chunking Guide

## Overview

**Sentence-window chunking** creates small, precise chunks for matching while storing surrounding context for retrieval expansion.

### The Problem It Solves

**Traditional chunking:**
```
Chunk (450 tokens): "...various policies including overtime, leave, sick pay, 
                     bonuses, pension, and other benefits..."
```
❌ Too broad - hard to match precise queries

**Sentence-window solution:**
```
Core chunk (50 tokens): "Overtime is paid at 1.5x normal rate"
Expanded (120 tokens): "Base salary is paid monthly. Overtime is paid at 1.5x 
                        normal rate. You must get approval before overtime."
```
✅ Precise matching + sufficient context

## How It Works

### Step 1: Split into Sentences
```
Document: "The company offers benefits. Base salary is paid on 25th. 
           Overtime is 1.5x. Approval needed. Bonuses are annual."

Sentences: [
    "The company offers benefits.",
    "Base salary is paid on 25th.",
    "Overtime is 1.5x.",
    "Approval needed.",
    "Bonuses are annual."
]
```

### Step 2: Group into Small Chunks
```
Target: 50 tokens per chunk

Chunk 1: "The company offers benefits. Base salary is paid on 25th."
Chunk 2: "Overtime is 1.5x. Approval needed."
Chunk 3: "Bonuses are annual."
```

### Step 3: Store Window Metadata
```python
{
    'text': 'Overtime is 1.5x. Approval needed.',  # Small chunk
    'metadata': {
        'sentence_start_idx': 2,  # Starts at sentence 2
        'sentence_end_idx': 3,    # Ends at sentence 3
        'window_start_idx': 1,    # Window includes sentence 1
        'window_end_idx': 4,      # Window includes sentence 4
        'expanded_text': 'Base salary is paid on 25th. Overtime is 1.5x. 
                         Approval needed. Bonuses are annual.',
        'estimated_tokens': 8,           # Core chunk
        'estimated_tokens_expanded': 16   # With window
    }
}
```

### Step 4: Retrieval Expansion
```
Query: "What is overtime rate?"

1. Search small chunks → Finds "Overtime is 1.5x"
2. Retrieve metadata → Get expanded_text
3. Return with context → "Base salary... Overtime is 1.5x... Approval needed..."
```

## Configuration

### Environment Variables

```bash
# .env
CHUNKING_METHOD=sentence-window
SENTENCE_WINDOW_CHUNK_TOKENS=50    # Core chunk size
SENTENCE_WINDOW_SENTENCES=2        # Context window (±2 sentences)
```

### Programmatic Usage

```python
from ingestion.sentence_window_chunking import chunk_text_sentence_window_with_metadata

# Create chunks
chunks = chunk_text_sentence_window_with_metadata(
    text=document_text,
    chunk_tokens=50,
    window_sentences=2
)

# Examine structure
for chunk in chunks:
    print(f"Core: {chunk['text']}")
    print(f"Expanded: {chunk['metadata']['expanded_text']}")
    print(f"Tokens: {chunk['metadata']['estimated_tokens']} → "
          f"{chunk['metadata']['estimated_tokens_expanded']}")
```

## Retrieval with Expansion

### Using the Sentence-Window Retriever

```python
from src.adapters.sentence_window_retriever_adapter import SentenceWindowMemoryRetriever

# Create retriever
retriever = SentenceWindowMemoryRetriever(
    memory_store=memory_store,
    ctx=ctx,
    top_k=5
)

# Search - automatically expands
docs = retriever.get_relevant_documents("What is the overtime rate?")

# Result includes context
print(docs[0].page_content)
# Output: "Base salary is paid on the 25th. Overtime is paid at 1.5x normal 
#          rate. You must get approval before working overtime."

# Check if expanded
print(docs[0].metadata['expanded'])  # True
print(docs[0].metadata['expansion_info'])  # "Expanded from 1 to 3 sentences"
```

## Parameter Tuning

### Chunk Size (tokens)

| Size | Use Case | Precision | Context |
|------|----------|-----------|---------|
| **30-40** | Very specific facts | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **50-70** | Balanced (recommended) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **80-100** | Longer answers | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Window Size (sentences)

| Window | Total Context | Best For |
|--------|---------------|----------|
| **0** | No expansion | Single-sentence facts |
| **1** | ±1 sentence | Short facts |
| **2** | ±2 sentences (recommended) | Most Q&A |
| **3** | ±3 sentences | Complex questions |
| **4-5** | ±4-5 sentences | Explanations |

## Benefits

### 1. Precision Matching
```
Query: "overtime rate"
Matches: Small chunk containing exactly "overtime rate"
Not: Large chunk with overtime + 10 other topics
```

### 2. Automatic Context
```
No need to:
- Retrieve multiple chunks manually
- Concatenate results
- Filter overlapping content

Just retrieve and expand automatically!
```

### 3. Efficient Storage
```
vs Multi-size chunking:
- Multi-size: 4x storage (50, 100, 200, 400 tokens)
- Sentence-window: 1x storage (50 tokens + metadata)
```

### 4. Better Than Pure Small Chunks
```
Pure small (50 tokens):
- ✅ Precise matching
- ❌ Lacks context
- ❌ Fragments incomplete

Sentence-window:
- ✅ Precise matching
- ✅ Automatic context
- ✅ Complete thoughts
```

## Comparison with Other Methods

### vs Token Chunking (450 tokens)

**Token:**
```
Chunk: "The company provides various benefits including base salary paid on 
        the 25th, overtime at 1.5x rate with approval, annual bonuses in 
        December, 26 days leave, sick pay coverage..."
```
❌ Query "overtime rate" retrieves chunk with irrelevant info (leave, bonuses)

**Sentence-window:**
```
Core: "Overtime is paid at 1.5x normal rate."
Expanded: "Base salary paid monthly. Overtime is paid at 1.5x normal rate. 
           You must get approval before overtime."
```
✅ Query "overtime rate" retrieves focused, relevant content

### vs Semantic Chunking

**Semantic:**
- Best for: Structured documents with clear sections
- Chunk size: Variable (section-dependent)
- Strength: Never mixes topics

**Sentence-window:**
- Best for: Factual Q&A, precise queries
- Chunk size: Small with expansion
- Strength: Maximum precision

**Recommendation:** Use both!
- Semantic for document structure
- Sentence-window for facts within sections

## Real-World Example

### Bank Handbook: Overtime Policy

**Document text:**
```
4.2 Overtime and Inconvenience Payments

The company recognizes the need for occasional overtime work. Overtime is paid 
at one and a half times your normal hourly rate. You must obtain prior approval 
from your line manager before working any overtime hours. Overtime is capped at 
10 hours per week unless exceptional circumstances apply.
```

**Token chunking (450 tokens):**
```
Chunk includes entire section 4.2 + part of 4.3 (bonuses)
→ Mixed content
```

**Semantic chunking:**
```
Chunk = complete section 4.2
→ Good, but still ~100 tokens
```

**Sentence-window (50 tokens, ±2 sentences):**
```
Chunk 1 (core): "The company recognizes the need for occasional overtime work."
Expanded: "4.2 Overtime and Inconvenience Payments. The company recognizes the 
           need for occasional overtime work. Overtime is paid at one and a half 
           times your normal hourly rate."

Chunk 2 (core): "Overtime is paid at one and a half times your normal hourly rate."
Expanded: "The company recognizes overtime work. Overtime is paid at one and a 
           half times your normal hourly rate. You must obtain prior approval 
           from your line manager."

Chunk 3 (core): "You must obtain prior approval from your line manager before 
                 working any overtime hours."
Expanded: "Overtime is paid at 1.5x. You must obtain prior approval from your 
           line manager before working any overtime hours. Overtime is capped 
           at 10 hours per week."
```

**Query:** "What is the overtime rate?"
- Token: Retrieves mixed content ❌
- Semantic: Retrieves full section (100 tokens) ✓
- Sentence-window: Retrieves Chunk 2 expanded (perfect match!) ✅

## Best Practices

1. **Start with defaults** (50 tokens, ±2 sentences)
2. **Adjust based on needs:**
   - More precision? → Smaller chunks (30-40 tokens)
   - More context? → Larger window (±3-4 sentences)
3. **Test with real queries** using `test_retrieval.py`
4. **Monitor performance:**
   - Are answers too fragmented? → Increase window
   - Getting irrelevant context? → Decrease window
5. **Combine with semantic** for structured documents

## Testing

```bash
# Test sentence-window chunking
CHUNKING_METHOD=sentence-window \
SENTENCE_WINDOW_CHUNK_TOKENS=50 \
SENTENCE_WINDOW_SENTENCES=2 \
poetry run python scripts/test_retrieval.py \
  --docs docs/ \
  --save-chunks sentence_window_chunks.txt \
  --query "What is overtime rate?"

# Review generated chunks
cat sentence_window_chunks.txt

# Look for:
# - Small core chunks (50 tokens)
# - expanded_text in metadata
# - Proper sentence boundaries
```

## Limitations

1. **Sentence detection** - relies on regex, may miss:
   - Unusual punctuation
   - Abbreviations (Mr., Dr., etc.)
   - Lists without periods

2. **Fixed window** - doesn't consider:
   - Semantic boundaries
   - Section changes
   - Topic shifts

3. **Best for Q&A** - not ideal for:
   - Long explanations
   - Multi-step processes
   - Comparative analysis

## When to Use

✅ **Use sentence-window when:**
- Building Q&A systems
- Extracting specific facts
- Answering "what", "how much", "when" questions
- Need maximum precision

❌ **Don't use when:**
- Documents lack clear sentences
- Need full section context (use semantic)
- Answers require multi-paragraph explanations

## Summary

**Sentence-window chunking** = Precision + Context

- Small chunks (50 tokens) for accurate matching
- Stored context (±N sentences) for comprehension
- Automatic expansion at retrieval
- Best for factual Q&A systems

**Result:** Better answer quality with efficient storage! 🎯


