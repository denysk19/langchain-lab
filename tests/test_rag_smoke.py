"""
Smoke test for basic RAG functionality.
Tests document loading, indexing, and query processing without console interaction.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import functions from chat script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from chat import (
    build_rag_graph,
    create_llm_provider,
    format_docs,
    load_documents,
)


@pytest.fixture
def temp_docs_dir():
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple markdown file with known content
        doc_path = Path(tmpdir) / "test_policy.md"
        doc_content = """# Test Company Policy

## Vacation Policy

Employees receive 25 vacation days per year. 
Vacation requests must be submitted 2 weeks in advance.
Unused vacation days expire at year end.

## Remote Work

Remote work is allowed up to 4 days per week.
All remote workers must attend weekly team meetings.
"""
        doc_path.write_text(doc_content)
        
        # Create a text file
        txt_path = Path(tmpdir) / "guidelines.txt"
        txt_content = """TESTING GUIDELINES

All code must include unit tests.
Code coverage should be above 90%.
Use pytest for testing framework.
"""
        txt_path.write_text(txt_content)
        
        yield tmpdir


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "sk-test-key-123",
        "LLM_PROVIDER": "openai",
        "MODEL": "gpt-4o-mini",
        "EMBED_MODEL": "text-embedding-3-small"
    }
    
    with patch.dict(os.environ, env_vars):
        yield


def test_load_documents(temp_docs_dir):
    """Test document loading from directory."""
    documents = load_documents(temp_docs_dir)
    
    assert len(documents) == 2
    
    # Check that documents have content and metadata
    for doc in documents:
        assert isinstance(doc, Document)
        assert len(doc.page_content) > 0
        assert "source" in doc.metadata
        assert doc.metadata["source"].endswith((".md", ".txt"))


def test_format_docs():
    """Test document formatting for context."""
    docs = [
        Document(
            page_content="This is the first document with important information.",
            metadata={"source": "/path/to/doc1.md"}
        ),
        Document(
            page_content="This is the second document with more details.",
            metadata={"source": "/path/to/doc2.txt"}
        )
    ]
    
    formatted = format_docs(docs)
    
    # Check formatting
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "SOURCE: /path/to/doc1.md" in formatted
    assert "SOURCE: /path/to/doc2.txt" in formatted
    assert "first document" in formatted
    assert "second document" in formatted


@patch('langchain_openai.ChatOpenAI')
def test_create_llm_provider_openai(mock_chat_openai, mock_env_vars):
    """Test OpenAI provider creation."""
    mock_llm = mock_chat_openai.return_value
    
    llm = create_llm_provider("openai", "gpt-4o-mini")
    
    # Verify ChatOpenAI was called with correct parameters
    mock_chat_openai.assert_called_once_with(
        model="gpt-4o-mini",
        api_key="sk-test-key-123",
        temperature=0
    )
    assert llm == mock_llm


@patch('langchain_openai.ChatOpenAI')
def test_create_llm_provider_vllm(mock_chat_openai):
    """Test vLLM provider creation."""
    env_vars = {
        "VLLM_BASE_URL": "http://localhost:8000",
        "VLLM_API_KEY": "sk-local",
        "VLLM_MODEL": "llama-3-8b"
    }
    
    with patch.dict(os.environ, env_vars):
        mock_llm = mock_chat_openai.return_value
        
        llm = create_llm_provider("vllm", "gpt-4o-mini")
        
        # Verify ChatOpenAI was called with vLLM configuration
        mock_chat_openai.assert_called_once_with(
            model="llama-3-8b",  # Should use VLLM_MODEL
            base_url="http://localhost:8000",
            api_key="sk-local",
            temperature=0
        )
        assert llm == mock_llm


def test_create_llm_provider_missing_key():
    """Test error handling for missing API keys."""
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        create_llm_provider("openai", "gpt-4o-mini")


def test_create_llm_provider_missing_vllm_url():
    """Test error handling for missing vLLM URL."""
    with pytest.raises(ValueError, match="VLLM_BASE_URL is required"):
        create_llm_provider("vllm", "gpt-4o-mini")


def test_create_llm_provider_invalid_provider(mock_env_vars):
    """Test error handling for invalid provider."""
    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        create_llm_provider("invalid", "gpt-4o-mini")


@patch('langchain_openai.OpenAIEmbeddings')
@patch('langchain_community.vectorstores.FAISS.from_documents')
@patch('langchain_openai.ChatOpenAI')
def test_rag_chain_integration(mock_chat_openai, mock_faiss, mock_embeddings, temp_docs_dir, mock_env_vars):
    """Integration test for the complete RAG pipeline."""
    
    # Mock components
    mock_llm = mock_chat_openai.return_value
    mock_llm.invoke.return_value.content = "Based on the documents, employees receive 25 vacation days per year [1]."
    
    mock_vectorstore = mock_faiss.return_value
    mock_retriever = mock_vectorstore.as_retriever.return_value
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Employees receive 25 vacation days per year. Vacation requests must be submitted 2 weeks in advance.",
            metadata={"source": f"{temp_docs_dir}/test_policy.md"}
        )
    ]
    mock_retriever.get_relevant_documents.return_value = test_docs
    
    # Load documents and create chain
    documents = load_documents(temp_docs_dir)
    assert len(documents) == 2
    
    # Mock text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = text_splitter.split_documents(documents)
    
    # Create LLM and graph
    llm = create_llm_provider("openai", "gpt-4o-mini")
    graph = build_rag_graph(llm, mock_retriever, k=4)
    
    # Test query without history
    initial_state = {
        "messages": [],
        "question": "How many vacation days do employees get?",
        "rewritten_query": "",
        "context": "",
        "answer": ""
    }
    
    result = graph.invoke(initial_state)
    response = result["answer"]
    
    # Verify response
    assert "25 vacation days" in response
    assert "[1]" in response
    
    # Verify retriever was called
    mock_retriever.get_relevant_documents.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])

