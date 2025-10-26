# Sentence-Window Chunking: Metadata Preservation Fix

## Problem

Sentence-window chunks were being **created correctly**, but the `expanded_text` metadata was **not being saved** to the database. This meant:

- ❌ Only small chunks were stored (no expanded context)
- ❌ `chunk_method` showed as "unknown" instead of "sentence-window"
- ❌ Retrieval couldn't display the expanded context
- ❌ LLM received only small chunks (50 tokens) instead of expanded context

## Root Cause

The issue was in how `ingest.py` handled different chunking methods:

```python
# OLD CODE (broken):
parts = chunk_text(text, chunk_size, overlap)
# ↑ This returns List[str] - just plain strings
# ↑ For sentence-window, metadata with expanded_text was lost!

vectors = embed_texts(parts)

for i, (t, v) in enumerate(zip(parts, vectors)):
    chunks.append(Chunk(text=t, metadata={...}, embedding=v))
    # ↑ No expanded_text in metadata!
```

The `chunk_text()` function returns only strings, not metadata. For sentence-window chunks, the metadata containing `expanded_text` was generated but immediately discarded.

## The Fix

### 1. Modified `src/ingestion/src/ingestion/ingest.py`

**Detection and Routing:**
```python
# NEW CODE (fixed):
if config.chunking_method == "sentence-window":
    from .sentence_window_chunking import chunk_text_sentence_window_with_metadata
    
    # Get chunks WITH metadata
    parts_with_metadata = chunk_text_sentence_window_with_metadata(
        text,
        chunk_tokens=config.sentence_window_chunk_tokens,
        window_sentences=config.sentence_window_sentences
    )
    
    # Extract small text for embedding (only 50 tokens)
    parts = [chunk['text'] for chunk in parts_with_metadata]
    
    # Store metadata separately (includes expanded_text!)
    chunk_metadata_list = [chunk['metadata'] for chunk in parts_with_metadata]
else:
    # Other methods (token, char, semantic)
    parts = chunk_text(text, chunk_size, overlap)
    chunk_metadata_list = [{}] * len(parts)
```

**Metadata Preservation:**
```python
# Embed only the small chunks
vectors = embed_texts(parts)

# Build chunks with preserved metadata
for i, (t, v, chunk_meta) in enumerate(zip(parts, vectors, chunk_metadata_list)):
    base_metadata = {
        "filename": filename,
        "chunk_method": config.chunking_method,
        # ... other base fields ...
    }
    # Merge with chunk-specific metadata (includes expanded_text!)
    base_metadata.update(chunk_meta)
    
    chunks.append(Chunk(
        text=t,              # Small chunk (50 tokens)
        metadata=base_metadata,  # Now includes expanded_text!
        embedding=v          # Embedding of small chunk
    ))
```

### 2. Modified `scripts/test_retrieval.py`

**Fixed Metadata Key:**
```python
# OLD: Incorrect key name
is_sentence_window = metadata.get('chunking_method') == 'sentence-window'

# NEW: Correct key with fallback
chunking_method = metadata.get('chunk_method', metadata.get('chunking_method', 'unknown'))
is_sentence_window = chunking_method == 'sentence-window'
```

The metadata key is `chunk_method` (from `ingest.py`), not `chunking_method`. Added fallback for backward compatibility.

## How It Works Now

### Step 1: Chunking (Creates Both)
```
Input: "Schedule 1: Core Leave. You are entitled to 26 working days' core leave. 
        This leave must be taken during the benefits year."

Output:
- Small chunk:    "You are entitled to 26 working days' core leave."
- Expanded text:  "Schedule 1: Core Leave. You are entitled to 26 working 
                   days' core leave. This leave must be taken during the 
                   benefits year."
```

### Step 2: Embedding (Small Chunks Only)
```python
embed_texts(["You are entitled to 26 working days' core leave."])
# ✅ Vector represents the PRECISE core chunk (50 tokens)
# ✅ Better matching for specific queries
```

### Step 3: Storage (Metadata Preserved)
```python
{
    "text": "You are entitled to 26 working days' core leave.",
    "metadata": {
        "chunk_method": "sentence-window",
        "expanded_text": "Schedule 1: Core Leave. You are entitled to...",
        "sentences_in_chunk": 2,
        "sentences_in_window": 4,
        "estimated_tokens": 48,
        "estimated_tokens_expanded": 95
    },
    "embedding": [0.123, -0.456, ...]  # Embedding of small chunk
}
```

### Step 4: Retrieval (Automatic Expansion)
```
Query: "How many days of core leave?"

1. Search matches against small chunks (precise matching)
   └─ Vector similarity: 0.85

2. Retrieve chunk with metadata
   └─ text: "You are entitled to 26 working days' core leave."
   └─ metadata.expanded_text: "Schedule 1: Core Leave. You are..."

3. Display shows BOTH:
   📌 CORE CHUNK: "You are entitled to 26 working days' core leave."
   🔍 EXPANDED CONTEXT: "Schedule 1: Core Leave. You are entitled to..."

4. LLM receives expanded context for better answer
```

## Benefits

✅ **Precise Matching**: Small chunks (50 tokens) = better vector similarity for specific queries  
✅ **Rich Context**: Expanded text (±2 sentences) = LLM has enough context to answer  
✅ **Single-Topic Chunks**: Each chunk focuses on one concept  
✅ **Automatic Expansion**: Happens transparently at retrieval time  
✅ **Cost Effective**: Only small chunks embedded (fewer tokens = lower cost)  

## Testing

### 1. Verify Configuration
```bash
# Check your .env file
cat .env | grep -A 3 CHUNKING_METHOD
```

Should show:
```bash
CHUNKING_METHOD=sentence-window
SENTENCE_WINDOW_CHUNK_TOKENS=50
SENTENCE_WINDOW_SENTENCES=2
```

### 2. Re-Ingest Documents
```bash
poetry run python scripts/test_retrieval.py \
  --docs docs/ \
  --k 5 \
  --query "How many days of core leave?"
```

### 3. Expected Output
```
📄 Processing: bank_handbook.pdf
   Method: Sentence-window chunking
   Chunk size: 50 tokens
   Window: ±2 sentences
✅ Created 287 sentence-window chunks
   └─ Avg core: 245 chars, Avg expanded: 589 chars

🔢 RANK #1
────────────────────────────────────────
🔧 Chunking Method: sentence-window  ← Should say this!
────────────────────────────────────────
📌 CORE CHUNK (2 sentences, ~48 tokens):
────────────────────────────────────────
  You are entitled to 26 working days' core leave

────────────────────────────────────────
🔍 EXPANDED CONTEXT (4 sentences, ~95 tokens):
────────────────────────────────────────
  Schedule 1: Core Leave. You are entitled to 26 
  working days' core leave each year. This leave 
  must be taken during the benefits year.
```

## Files Changed

1. **`src/ingestion/src/ingestion/ingest.py`**
   - Added sentence-window detection
   - Call `chunk_text_sentence_window_with_metadata()`
   - Preserve metadata including `expanded_text`
   - Updated progress messages

2. **`scripts/test_retrieval.py`**
   - Fixed metadata key: `chunk_method` (not `chunking_method`)
   - Added fallback for backward compatibility
   - Updated display logic for both functions

## Comparison: Before vs After

| Aspect | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| **Embedding** | ✅ Small chunks only | ✅ Small chunks only |
| **Metadata Storage** | ❌ Lost `expanded_text` | ✅ Preserved `expanded_text` |
| **Chunk Method** | ❌ Shows "unknown" | ✅ Shows "sentence-window" |
| **Display** | ❌ Only small text | ✅ Both core + expanded |
| **LLM Context** | ❌ 50 tokens (too small) | ✅ 95 tokens (sufficient) |
| **Retrieval Quality** | ✅ Precise matching | ✅ Precise matching |
| **Answer Quality** | ❌ Poor (not enough context) | ✅ Good (expanded context) |

## Next Steps

1. ✅ **Configuration**: Ensure `.env` has sentence-window settings
2. ✅ **Re-Ingestion**: Run `test_retrieval.py` to re-ingest documents
3. ✅ **Verification**: Check that `chunk_method` shows "sentence-window"
4. ✅ **Testing**: Verify expanded context is displayed
5. ⏭️ **Integration**: Update `chat.py` to use `SentenceWindowMemoryRetriever`
6. ⏭️ **Evaluation**: Run `evaluate.py` to measure RAG performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PDF Document                                               │
└───────────────────────────────┬─────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  sentence_window_      │
                    │  chunking.py           │
                    └───────────┬────────────┘
                                │
                ┌───────────────▼──────────────────┐
                │  Small Chunk + Expanded Context  │
                │  ┌──────────────────────────┐    │
                │  │ text: "You are..."       │    │
                │  │ metadata:                │    │
                │  │   expanded_text: "..."   │    │
                │  └──────────────────────────┘    │
                └───────────────┬──────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Embedder              │
                    │  (only small chunk)    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  MemoryStore           │
                    │  (with expanded_text)  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Retrieval             │
                    │  (returns expanded)    │
                    └───────────┬────────────┘
                                │
                        ┌───────▼──────┐
                        │  LLM         │
                        │  (full ctx)  │
                        └──────────────┘
```

## Credits

- **Implementation**: Sentence-window chunking with metadata preservation
- **Fix Applied**: October 20, 2025
- **Files Modified**: 2 (ingest.py, test_retrieval.py)
- **Lines Changed**: ~80 lines
- **Status**: ✅ Ready for testing


