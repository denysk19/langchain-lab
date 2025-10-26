"""Memory retriever adapter with sentence-window expansion support.

This adapter extends the basic MemoryRetrieverAdapter to support sentence-window chunks.
When retrieving chunks created with sentence-window chunking, it automatically expands
them to include the surrounding context (±N sentences).
"""

from typing import List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ingestion import MemoryStore, DocumentCtx, SearchHit


class SentenceWindowMemoryRetriever(BaseRetriever):
    """
    LangChain retriever that uses MemoryStore and expands sentence-window chunks.
    
    Features:
    - Retrieves small, precise chunks for accurate matching
    - Automatically expands to include context window (±N sentences)
    - Falls back to regular chunks if no sentence-window metadata
    - Compatible with all chunking methods
    
    Example:
        >>> retriever = SentenceWindowMemoryRetriever(
        ...     memory_store=store,
        ...     ctx=ctx,
        ...     top_k=5
        ... )
        >>> docs = retriever.get_relevant_documents("What is overtime rate?")
        >>> # Returns expanded chunks with context
    """
    
    memory_store: MemoryStore
    ctx: DocumentCtx
    top_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve and expand chunks based on their type.
        
        For sentence-window chunks: returns expanded text with context
        For other chunks: returns text as-is
        """
        # Get search results from memory store
        hits: List[SearchHit] = self.memory_store.search(
            ctx=self.ctx,
            query=query,
            top_k=self.top_k
        )
        
        # Convert to LangChain documents with expansion
        docs = []
        for hit in hits:
            metadata = hit.metadata.copy()
            
            # Check if chunk has sentence-window metadata
            if 'expanded_text' in metadata and 'chunking_method' in metadata:
                if metadata['chunking_method'] == 'sentence-window':
                    # Use expanded text for better context
                    page_content = metadata['expanded_text']
                    metadata['chunk_text'] = hit.text  # Keep original for reference
                    metadata['expanded'] = True
                    metadata['expansion_info'] = (
                        f"Expanded from {metadata.get('sentences_in_chunk', 'N/A')} "
                        f"to {metadata.get('sentences_in_window', 'N/A')} sentences"
                    )
                else:
                    # Other chunking method, use as-is
                    page_content = hit.text
                    metadata['expanded'] = False
            else:
                # Regular chunk without sentence-window metadata, use as-is
                page_content = hit.text
                metadata['expanded'] = False
            
            # Add search metadata
            metadata.update({
                'document_id': hit.document_id,
                'chunk_index': hit.chunk_index,
                'score': hit.score,
            })
            
            docs.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        return docs


def create_sentence_window_retriever(
    memory_store: MemoryStore,
    ctx: DocumentCtx,
    top_k: int = 5
) -> SentenceWindowMemoryRetriever:
    """
    Convenience function to create a sentence-window retriever.
    
    Args:
        memory_store: MemoryStore instance
        ctx: Document context
        top_k: Number of results to retrieve
        
    Returns:
        Configured retriever with sentence-window expansion
        
    Example:
        >>> from src.adapters.sentence_window_retriever_adapter import create_sentence_window_retriever
        >>> retriever = create_sentence_window_retriever(store, ctx, top_k=5)
        >>> docs = retriever.get_relevant_documents("query")
    """
    return SentenceWindowMemoryRetriever(
        memory_store=memory_store,
        ctx=ctx,
        top_k=top_k
    )


