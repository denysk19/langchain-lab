#!/usr/bin/env python3
"""
Test script to verify ingestion module integration.
This script verifies that all components are properly installed and configured.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    # Test ingestion module
    try:
        from ingestion.models import DocumentCtx
        from ingestion.ingest import ingest_pdf
        from ingestion.memory_store import MemoryStore
        print("✅ Ingestion module imports successful")
    except ImportError as e:
        print(f"❌ Failed to import ingestion module: {e}")
        print("   → Install with: cd ingestion/ && pip install -e .")
        return False
    
    # Test adapter
    try:
        from src.adapters import MemoryRetrieverAdapter
        print("✅ MemoryRetrieverAdapter import successful")
    except ImportError as e:
        print(f"❌ Failed to import MemoryRetrieverAdapter: {e}")
        return False
    
    # Test RAG workflow
    try:
        from rag_workflow import create_rag_workflow, RAGState
        print("✅ RAG workflow imports successful")
    except ImportError as e:
        print(f"❌ Failed to import RAG workflow: {e}")
        print("   → Ensure rag-module submodule is initialized")
        return False
    
    # Test LangChain
    try:
        from langchain_core.documents import Document
        from langchain_openai import ChatOpenAI
        print("✅ LangChain imports successful")
    except ImportError as e:
        print(f"❌ Failed to import LangChain: {e}")
        print("   → Install with: poetry install")
        return False
    
    return True


def test_adapter():
    """Test that the adapter works correctly."""
    print("\nTesting MemoryRetrieverAdapter...")
    
    try:
        from ingestion.models import DocumentCtx
        from ingestion.memory_store import MemoryStore
        from src.adapters import MemoryRetrieverAdapter
        from langchain_core.documents import Document
        
        # Create a MemoryStore
        store = MemoryStore()
        
        # Create a DocumentCtx
        ctx = DocumentCtx(
            tenant_id="test-tenant",
            owner_user_id="test-user",
            document_id="test-doc",
            visibility="org",
            embedding_version="openai:text-embedding-3-small@v1"
        )
        
        # Create adapter
        adapter = MemoryRetrieverAdapter(store=store, ctx=ctx, top_k=5)
        print("✅ MemoryRetrieverAdapter instantiation successful")
        
        # Test retrieval interface (will return empty results)
        results = adapter.get_relevant_documents("test query")
        assert isinstance(results, list), "get_relevant_documents should return a list"
        print(f"✅ get_relevant_documents returned {len(results)} documents (expected 0 for empty store)")
        
        return True
        
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_processing():
    """Test that PDF processing works (if sample PDF exists)."""
    print("\nTesting PDF processing...")
    
    # Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment")
        print("   Create .env file with: OPENAI_API_KEY=sk-...")
        print("   Skipping PDF processing test (requires OpenAI API)")
        return True
    
    # Check if sample PDF exists
    sample_pdf = Path("docs/bank_handbook.pdf")
    if not sample_pdf.exists():
        print("⚠️  No sample PDF found at docs/bank_handbook.pdf")
        print("   Skipping PDF processing test")
        return True
    
    try:
        from ingestion.models import DocumentCtx
        from ingestion.ingest import ingest_pdf
        from ingestion.memory_store import MemoryStore
        
        # Create store
        store = MemoryStore()
        
        # Create context
        ctx = DocumentCtx(
            tenant_id="test-tenant",
            owner_user_id="test-user",
            document_id="test-pdf",
            visibility="org",
            embedding_version="openai:text-embedding-3-small@v1"
        )
        
        # Read PDF
        with open(sample_pdf, "rb") as f:
            pdf_bytes = f.read()
        
        print(f"   Processing {sample_pdf.name} ({len(pdf_bytes)} bytes)...")
        
        # Ingest (this will call OpenAI API if OPENAI_API_KEY is set)
        result = ingest_pdf(
            ctx=ctx,
            filename=sample_pdf.name,
            raw_pdf_bytes=pdf_bytes,
            sink=store,
            chunk_chars=1000,
            overlap=150
        )
        
        chunks_saved = result['chunks_saved']
        print(f"✅ PDF processing successful: {chunks_saved} chunks saved")
        print(f"   Content hash: {result['content_hash']}")
        print(f"   Embedding version: {result['embedding_version']}")
        
        # Test retrieval
        from src.adapters import MemoryRetrieverAdapter
        adapter = MemoryRetrieverAdapter(store=store, ctx=ctx, top_k=3)
        results = adapter.get_relevant_documents("policy")
        print(f"✅ Retrieval successful: found {len(results)} relevant chunks")
        
        if results:
            print(f"   Sample result: {results[0].page_content[:100]}...")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("INGESTION MODULE INTEGRATION TEST")
    print("=" * 70)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test adapter
    if results[0][1]:  # Only run if imports passed
        results.append(("Adapter", test_adapter()))
        
        # Test PDF processing (requires OPENAI_API_KEY)
        results.append(("PDF Processing", test_pdf_processing()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Integration successful.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY in .env")
        print("2. Add PDF documents to docs/")
        print("3. Run: python scripts/chat.py --docs docs/")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

