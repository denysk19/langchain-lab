"""Utility functions for RAG workflow."""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


def format_docs(docs: List[Document]) -> str:
    """Format documents for context with citations."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:1000]  # First 1000 chars
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{i}] {content}\nSOURCE: {source}")
    
    return "\n\n".join(formatted)


def extract_conversation_context(messages: List[BaseMessage], max_messages: int = 6) -> str:
    """Extract recent conversation context from messages."""
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    if len(recent_messages) <= 1:
        return ""
    
    context_parts = []
    for msg in recent_messages[:-1]:  # Exclude current question
        if isinstance(msg, HumanMessage):
            context_parts.append(f"Employee: {msg.content}")
        elif isinstance(msg, AIMessage):
            context_parts.append(f"Assistant: {msg.content}")
    
    return "\n".join(context_parts) if context_parts else ""


def create_conversation_summary_prompt(messages: List[BaseMessage], current_summary: str = None) -> str:
    """Create a prompt for conversation summarization using PromptManager."""
    from .prompt_manager import PromptManager
    
    # Convert messages to text format for summarization
    conversation_text = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"Employee: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_text.append(f"Assistant: {msg.content}")
    
    conversation = "\n".join(conversation_text)
    
    # Use PromptManager for consistent prompt handling
    prompt_manager = PromptManager()
    return prompt_manager.get_summarization_prompt(conversation, current_summary)
