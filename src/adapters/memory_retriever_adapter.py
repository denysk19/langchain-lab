"""
LangChain adapter for MemoryStore retriever from ingestion module.
Bridges the ingestion module's Retriever protocol to LangChain's retriever interface.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


class MemoryRetrieverAdapter:
    """
    Adapter to use MemoryStore with LangChain.
    
    This adapter implements LangChain's BaseRetriever interface while using
    the ingestion module's MemoryStore as the underlying storage.
    
    Example:
        >>> from ingestion.memory_store import MemoryStore
        >>> from ingestion.models import DocumentCtx
        >>> 
        >>> store = MemoryStore()
        >>> ctx = DocumentCtx(
        ...     tenant_id="acme-corp",
        ...     owner_user_id="user-123",
        ...     document_id="doc-001",
        ...     visibility="org",
        ...     embedding_version="openai:text-embedding-3-small@v1"
        ... )
        >>> 
        >>> retriever = MemoryRetrieverAdapter(store, ctx, top_k=5)
        >>> docs = retriever.get_relevant_documents("What is the refund policy?")
    """

    def __init__(self, store, ctx, top_k: int = 8):
        """
        Initialize the adapter.
        
        Args:
            store: MemoryStore instance from ingestion module
            ctx: DocumentCtx instance with tenant/document context
            top_k: Number of documents to retrieve (default: 8)
        """
        self.store = store
        self.ctx = ctx
        self.top_k = top_k

    def get_relevant_documents(
        self, 
        query: str,
        callbacks: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: The search query string
            callbacks: Optional LangChain callback manager (not used)
            
        Returns:
            List of LangChain Document objects with content and metadata
        """
        # Use the ingestion module's search method
        hits = self.store.search(ctx=self.ctx, query=query, top_k=self.top_k)
        
        # Convert SearchResult objects to LangChain Documents
        documents = []
        for hit in hits:
            # Merge hit metadata with additional fields
            metadata = hit.metadata.copy() if hit.metadata else {}
            metadata.update({
                "document_id": hit.document_id,
                "chunk_index": hit.chunk_index,
                "score": hit.score,
            })
            
            # Create LangChain Document
            doc = Document(
                page_content=hit.text,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents

    async def aget_relevant_documents(
        self, 
        query: str,
        callbacks: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Async version of get_relevant_documents.
        
        Note: Currently just wraps the synchronous version.
        If the ingestion module adds async support, this should be updated.
        """
        return self.get_relevant_documents(query, callbacks)

    def _get_relevant_documents(
        self, 
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Internal method for LangChain BaseRetriever compatibility.
        """
        return self.get_relevant_documents(query)
    
    async def _aget_relevant_documents(
        self, 
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Internal async method for LangChain BaseRetriever compatibility.
        """
        return await self.aget_relevant_documents(query)

