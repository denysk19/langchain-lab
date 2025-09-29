"""RAG workflow state definition."""

from typing import List
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """State for the RAG workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    rewritten_query: str
    context: str
    answer: str
    needs_retrieval: bool
