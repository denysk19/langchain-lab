#!/usr/bin/env python3
"""
Interactive Retrieval Testing Script

Test document retrieval from MemoryStore with scoring and detailed chunk inspection.
Useful for debugging retrieval quality and understanding what gets retrieved.
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from tqdm import tqdm

# Import ingestion module
from ingestion import ingest_pdf, MemoryStore, DocumentCtx
from ingestion.config import get_config


def save_chunks_to_file(memory_store: MemoryStore, ctx: DocumentCtx, output_path: str):
    """
    Save all chunks from memory store to a text file.
    
    Args:
        memory_store: MemoryStore instance
        ctx: Document context
        output_path: Path to output file
    """
    print(f"💾 Saving chunks to {output_path}...")
    
    # Get the scope key to access raw data
    scope_key = memory_store._scope_key(ctx)
    
    # Get metadata for this scope
    metadata_list = memory_store.meta.get(scope_key, [])
    
    if not metadata_list:
        print("❌ No chunks found in memory store")
        return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHUNKS GENERATED DURING INGESTION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total chunks: {len(metadata_list)}\n")
        f.write(f"Scope: {scope_key}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, meta in enumerate(metadata_list, 1):
            doc_id = meta.get('document_id', 'Unknown')
            chunk_index = meta.get('chunk_index', 'N/A')
            text = meta.get('text', '')
            metadata = meta.get('metadata', {})
            
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            # Note: metadata key is 'chunk_method' from ingest.py
            chunking_method = metadata.get('chunk_method', metadata.get('chunking_method', 'unknown'))
            
            # Check if sentence-window chunk
            is_sentence_window = chunking_method == 'sentence-window'
            expanded_text = metadata.get('expanded_text')
            
            f.write("=" * 80 + "\n")
            f.write(f"CHUNK #{i}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Document ID: {doc_id}\n")
            f.write(f"Chunk Index: {chunk_index}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Page: {page}\n")
            f.write(f"Chunk ID: {doc_id}::{chunk_index}\n")
            f.write(f"Chunking Method: {chunking_method}\n")
            
            if is_sentence_window and expanded_text:
                # Sentence-window chunk: show both core and expanded
                sentences_in_chunk = metadata.get('sentences_in_chunk', 'N/A')
                sentences_in_window = metadata.get('sentences_in_window', 'N/A')
                estimated_tokens = metadata.get('estimated_tokens', 'N/A')
                estimated_tokens_expanded = metadata.get('estimated_tokens_expanded', 'N/A')
                
                f.write(f"Core Size: {len(text)} chars, {estimated_tokens} tokens, {sentences_in_chunk} sentences\n")
                f.write(f"Expanded Size: {len(expanded_text)} chars, {estimated_tokens_expanded} tokens, {sentences_in_window} sentences\n")
                f.write("-" * 80 + "\n")
                f.write("CORE CHUNK (what was embedded):\n")
                f.write("-" * 80 + "\n")
                f.write(text)
                f.write("\n\n")
                f.write("-" * 80 + "\n")
                f.write("EXPANDED CONTEXT (for retrieval):\n")
                f.write("-" * 80 + "\n")
                f.write(expanded_text)
                f.write("\n\n")
            else:
                # Regular chunk
                f.write(f"Text Length: {len(text)} characters\n")
                f.write("-" * 80 + "\n")
                f.write("CONTENT:\n")
                f.write("-" * 80 + "\n")
                f.write(text)
                f.write("\n\n")
    
    print(f"✅ Saved {len(metadata_list)} chunks to {output_path}")


def load_documents_into_memory(docs_path: str, ctx: DocumentCtx, 
                               chunk_size: int = None, 
                               overlap: int = None) -> tuple[MemoryStore, int]:
    """
    Load all PDF documents into MemoryStore.
    
    Returns:
        Tuple of (MemoryStore instance, total chunks ingested)
    """
    memory_store = MemoryStore()
    pdf_files = glob(os.path.join(docs_path, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"❌ No PDF documents found in {docs_path}")
        return memory_store, 0
    
    print(f"📚 Loading {len(pdf_files)} PDF files...")
    print()
    
    total_chunks = 0
    
    for file_path in tqdm(pdf_files, desc="Ingesting documents"):
        try:
            file_name = os.path.basename(file_path)
            doc_id = file_name.replace('.pdf', '').replace(' ', '_')
            
            doc_ctx = DocumentCtx(
                tenant_id=ctx.tenant_id,
                owner_user_id=ctx.owner_user_id,
                document_id=doc_id,
                visibility=ctx.visibility,
                embedding_version=ctx.embedding_version
            )
            
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            
            ingest_kwargs = {
                'ctx': doc_ctx,
                'filename': file_name,
                'raw_pdf_bytes': pdf_bytes,
                'sink': memory_store,
            }
            if chunk_size is not None:
                ingest_kwargs['chunk_size'] = chunk_size
            if overlap is not None:
                ingest_kwargs['overlap'] = overlap
            
            result = ingest_pdf(**ingest_kwargs)
            total_chunks += result['chunks_saved']
            
        except Exception as e:
            print(f"❌ Error ingesting {file_path}: {e}")
    
    print()
    print(f"✅ Ingested {total_chunks} total chunks from {len(pdf_files)} documents")
    print()
    
    return memory_store, total_chunks


def search_and_display(memory_store: MemoryStore, ctx: DocumentCtx, 
                       query: str, top_k: int = 5):
    """
    Search for query and display results with scores and metadata.
    """
    print()
    print("=" * 80)
    print(f"🔍 QUERY: {query}")
    print("=" * 80)
    print()
    
    # Search using the memory store's search method
    try:
        results = memory_store.search(
            ctx=ctx,
            query=query,
            top_k=top_k
        )
        
        if not results:
            print("❌ No results found")
            return
        
        print(f"📊 Retrieved {len(results)} chunks:")
        print()
        
        for i, result in enumerate(results, 1):
            # Extract data from SearchHit
            doc_id = result.document_id
            chunk_index = result.chunk_index
            score = result.score
            text = result.text
            metadata = result.metadata
            
            # Build chunk ID
            chunk_id = f"{doc_id}::{chunk_index}"
            
            # Check if this is a sentence-window chunk with expanded text
            # Note: metadata key is 'chunk_method' from ingest.py
            chunking_method_value = metadata.get('chunk_method', metadata.get('chunking_method', 'unknown'))
            is_sentence_window = chunking_method_value == 'sentence-window'
            has_expanded = 'expanded_text' in metadata
            
            if is_sentence_window and has_expanded:
                # Sentence-window: show both core and expanded
                core_text = text
                expanded_text = metadata['expanded_text']
                sentences_in_chunk = metadata.get('sentences_in_chunk', 'N/A')
                sentences_in_window = metadata.get('sentences_in_window', 'N/A')
                estimated_tokens = metadata.get('estimated_tokens', 'N/A')
                estimated_tokens_expanded = metadata.get('estimated_tokens_expanded', 'N/A')
            else:
                # Regular chunk
                core_text = None
                expanded_text = text
                sentences_in_chunk = None
                sentences_in_window = None
                estimated_tokens = None
                estimated_tokens_expanded = None
            
            # Extract useful metadata
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            chunking_method = chunking_method_value  # Already extracted above
            
            print(f"{'─' * 80}")
            print(f"🔢 RANK #{i}")
            print(f"{'─' * 80}")
            print(f"📄 Source: {source}")
            print(f"🆔 Document ID: {doc_id}")
            print(f"📑 Chunk ID: {chunk_id}")
            print(f"📊 Distance Score: {score:.4f} (lower is better)")
            print(f"📖 Page: {page} | Chunk Index: {chunk_index}")
            print(f"🔧 Chunking Method: {chunking_method}")
            
            if is_sentence_window and has_expanded:
                # Sentence-window chunk: show both core and expanded
                print(f"{'─' * 80}")
                print(f"📌 CORE CHUNK ({sentences_in_chunk} sentences, ~{estimated_tokens} tokens):")
                print(f"{'─' * 80}")
                
                # Display core text
                lines = core_text.split('\n')
                for line in lines:
                    if len(line) <= 76:
                        print(f"  {line}")
                    else:
                        words = line.split()
                        current_line = "  "
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 76:
                                current_line += word + " "
                            else:
                                print(current_line.rstrip())
                                current_line = "  " + word + " "
                        if current_line.strip():
                            print(current_line.rstrip())
                
                print()
                print(f"{'─' * 80}")
                print(f"🔍 EXPANDED CONTEXT ({sentences_in_window} sentences, ~{estimated_tokens_expanded} tokens):")
                print(f"{'─' * 80}")
                
                # Display expanded text
                lines = expanded_text.split('\n')
                for line in lines:
                    if len(line) <= 76:
                        print(f"  {line}")
                    else:
                        words = line.split()
                        current_line = "  "
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 76:
                                current_line += word + " "
                            else:
                                print(current_line.rstrip())
                                current_line = "  " + word + " "
                        if current_line.strip():
                            print(current_line.rstrip())
            else:
                # Regular chunk: show content as before
                print(f"{'─' * 80}")
                print(f"📝 CONTENT:")
                print(f"{'─' * 80}")
                
                lines = expanded_text.split('\n')
                for line in lines:
                    if len(line) <= 76:
                        print(f"  {line}")
                    else:
                        words = line.split()
                        current_line = "  "
                        for word in words:
                            if len(current_line) + len(word) + 1 <= 76:
                                current_line += word + " "
                            else:
                                print(current_line.rstrip())
                                current_line = "  " + word + " "
                        if current_line.strip():
                            print(current_line.rstrip())
            
            print()
        
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode(memory_store: MemoryStore, ctx: DocumentCtx, default_k: int):
    """
    Interactive query mode.
    """
    print()
    print("🎯 INTERACTIVE RETRIEVAL TESTING MODE")
    print("=" * 80)
    print("Commands:")
    print("  - Type a query to search")
    print("  - '/k <number>' to change top-k (current: {})".format(default_k))
    print("  - '/exit' or '/quit' to exit")
    print("=" * 80)
    print()
    
    current_k = default_k
    
    while True:
        try:
            user_input = input(f"query (k={current_k})> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/exit', '/quit']:
                print("👋 Goodbye!")
                break
            
            if user_input.startswith('/k '):
                try:
                    new_k = int(user_input.split()[1])
                    current_k = new_k
                    print(f"✅ Top-k set to {current_k}")
                except (ValueError, IndexError):
                    print("❌ Invalid format. Use: /k <number>")
                continue
            
            # Execute search
            search_and_display(memory_store, ctx, user_input, current_k)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive document retrieval testing tool"
    )
    
    # Document configuration
    parser.add_argument(
        "--docs",
        default="docs",
        help="Documents directory (default: docs)"
    )
    
    # Retrieval configuration
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    
    # Optional query for non-interactive mode
    parser.add_argument(
        "--query",
        help="Single query to test (non-interactive mode)"
    )
    
    # Save chunks option
    parser.add_argument(
        "--save-chunks",
        help="Save all generated chunks to a text file (e.g., chunks_output.txt)"
    )
    
    # Ingestion configuration
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (uses config default if not set)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap (uses config default if not set)"
    )
    
    # Document context configuration
    parser.add_argument(
        "--tenant-id",
        default="default-tenant",
        help="Tenant ID for multi-tenancy (default: default-tenant)"
    )
    parser.add_argument(
        "--owner-user-id",
        default="admin",
        help="Document owner user ID (default: admin)"
    )
    parser.add_argument(
        "--visibility",
        choices=["org", "private"],
        default="org",
        help="Document visibility scope (default: org)"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Get ingestion config
    ingestion_config = get_config()
    
    print()
    print("=" * 80)
    print("🔬 RETRIEVAL TESTING TOOL")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  📁 Documents: {args.docs}")
    print(f"  🔢 Embedder: {ingestion_config.get_embedding_model()}")
    print(f"  📏 Chunk Size: {args.chunk_size or ingestion_config.chunk_size}")
    print(f"  🔗 Chunk Overlap: {args.chunk_overlap or ingestion_config.chunk_overlap}")
    print(f"  🔝 Top-K: {args.k}")
    print()
    
    # Create document context
    doc_ctx = DocumentCtx(
        tenant_id=args.tenant_id,
        owner_user_id=args.owner_user_id,
        document_id="master",
        visibility=args.visibility,
        embedding_version=ingestion_config.get_embedding_model()
    )
    
    # Load documents
    memory_store, total_chunks = load_documents_into_memory(
        docs_path=args.docs,
        ctx=doc_ctx,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap
    )
    
    if total_chunks == 0:
        print("❌ No documents loaded. Exiting.")
        sys.exit(1)
    
    # Save chunks to file if requested
    if args.save_chunks:
        save_chunks_to_file(memory_store, doc_ctx, args.save_chunks)
        print()
    
    # Single query mode or interactive mode
    if args.query:
        search_and_display(memory_store, doc_ctx, args.query, args.k)
    else:
        interactive_mode(memory_store, doc_ctx, args.k)


if __name__ == "__main__":
    main()

