#!/usr/bin/env python3
"""
Interactive console RAG chatbot with persistent thread memory.
Supports OpenAI and vLLM providers via factory pattern.
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict
from tqdm import tqdm

# Enhanced PDF support with multiple fallback options
PDF_READERS = {}

# Try PyPDF2 first (most common)
try:
    from PyPDF2 import PdfReader as PyPDF2Reader
    PDF_READERS['pypdf2'] = PyPDF2Reader
except ImportError:
    pass

# Try pypdf (newer version)
try:
    from pypdf import PdfReader as PyPdfReader
    PDF_READERS['pypdf'] = PyPdfReader
except ImportError:
    pass

# Try pdfplumber (better for complex layouts)
try:
    import pdfplumber
    PDF_READERS['pdfplumber'] = pdfplumber
except ImportError:
    pass

# Try pymupdf (excellent performance)
try:
    import fitz  # PyMuPDF
    PDF_READERS['pymupdf'] = fitz
except ImportError:
    pass

HAS_PDF = bool(PDF_READERS)

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Default to less verbose
    format='%(message)s',   # Simpler format
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RAG")


class ThreadMetadata(BaseModel):
    """Metadata for thread management."""
    thread_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    message_count: int = 0
    title: Optional[str] = None
    tags: List[str] = []
    summary: Optional[str] = None  # Conversation summary
    last_summarized_count: int = 0  # Last message count when summary was updated


class EnhancedChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """Enhanced chat message history with metadata, size limits, and summarization."""
    
    messages: List[BaseMessage] = []
    metadata: ThreadMetadata
    max_messages: int = 20  # Reduced limit since we use summarization
    summarize_threshold: int = 12  # Start summarizing after this many messages
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self.metadata.message_count += 1
        self.metadata.last_accessed = datetime.now()
        
        # Check if we need to summarize and truncate
        if len(self.messages) > self.summarize_threshold:
            self._update_summary_if_needed()
            
        # Enforce message limit (keep recent messages)
        if len(self.messages) > self.max_messages:
            removed_count = len(self.messages) - self.max_messages
            self.messages = self.messages[-self.max_messages:]
            logger.debug(f"Removed {removed_count} old messages from thread {self.metadata.thread_id}")
    
    def _update_summary_if_needed(self) -> None:
        """Update conversation summary when needed."""
        messages_since_summary = len(self.messages) - self.metadata.last_summarized_count
        
        # Update summary every 8 new messages
        if messages_since_summary >= 8:
            try:
                # Get messages to summarize (exclude very recent ones)
                messages_to_summarize = self.messages[:-4] if len(self.messages) > 4 else self.messages[:-2]
                
                if messages_to_summarize:
                    new_summary = create_conversation_summary(
                        messages_to_summarize,
                        self.metadata.summary
                    )
                    
                    if new_summary:
                        self.metadata.summary = new_summary
                        self.metadata.last_summarized_count = len(self.messages) - 4
                        logger.info(f"Updated conversation summary for thread {self.metadata.thread_id}")
                        
            except Exception as e:
                logger.error(f"Failed to update summary: {e}")
    
    def clear(self) -> None:
        self.messages.clear()
        self.metadata.message_count = 0
        self.metadata.last_accessed = datetime.now()
        self.metadata.summary = None
        self.metadata.last_summarized_count = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get thread summary for management commands."""
        return {
            "thread_id": self.metadata.thread_id,
            "user_id": self.metadata.user_id,
            "title": self.metadata.title or f"Thread {self.metadata.thread_id[:8]}",
            "message_count": len(self.messages),
            "created_at": self.metadata.created_at.isoformat(),
            "last_accessed": self.metadata.last_accessed.isoformat(),
            "tags": self.metadata.tags
        }


class ThreadSafeMemoryManager:
    """Thread-safe memory manager for chat sessions."""
    
    def __init__(self, state_dir: str = ".state", max_threads_per_user: int = 10):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.max_threads_per_user = max_threads_per_user
        
        # Thread-safe caches
        self._history_cache: Dict[str, EnhancedChatMessageHistory] = {}
        self._user_threads: Dict[str, List[str]] = {}  # user_id -> [thread_ids]
        self._lock = threading.RLock()
        
        # Load existing thread mappings
        self._load_user_mappings()
    
    def _get_thread_file_path(self, thread_id: str) -> Path:
        """Get file path for thread storage."""
        return self.state_dir / f"thread_{thread_id}.json"
    
    def _get_user_mapping_file(self) -> Path:
        """Get file path for user->threads mapping."""
        return self.state_dir / "user_mappings.json"
    
    def _load_user_mappings(self) -> None:
        """Load user->thread mappings from disk."""
        mapping_file = self._get_user_mapping_file()
        if mapping_file.exists():
            try:
                with open(mapping_file, "r") as f:
                    self._user_threads = json.load(f)
                logger.debug(f"Loaded user mappings for {len(self._user_threads)} users")
            except Exception as e:
                logger.warning(f"Could not load user mappings: {e}")
                self._user_threads = {}
    
    def _save_user_mappings(self) -> None:
        """Save user->thread mappings to disk."""
        mapping_file = self._get_user_mapping_file()
        try:
            with open(mapping_file, "w") as f:
                json.dump(self._user_threads, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save user mappings: {e}")
    
    def get_or_create_thread(self, thread_id: str, user_id: str) -> EnhancedChatMessageHistory:
        """Get or create a thread for a user."""
        with self._lock:
            if thread_id in self._history_cache:
                # Update access time
                self._history_cache[thread_id].metadata.last_accessed = datetime.now()
                return self._history_cache[thread_id]
            
            # Try to load from disk
            thread_file = self._get_thread_file_path(thread_id)
            if thread_file.exists():
                history = self._load_thread_from_disk(thread_id, user_id)
                if history:
                    self._history_cache[thread_id] = history
                    return history
            
            # Create new thread
            return self._create_new_thread(thread_id, user_id)
    
    def _load_thread_from_disk(self, thread_id: str, user_id: str) -> Optional[EnhancedChatMessageHistory]:
        """Load thread from disk."""
        thread_file = self._get_thread_file_path(thread_id)
        try:
            with open(thread_file, "r") as f:
                data = json.load(f)
            
            # Create metadata
            metadata = ThreadMetadata(
                thread_id=thread_id,
                user_id=user_id,
                created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
                last_accessed=datetime.now(),
                message_count=data["metadata"]["message_count"],
                title=data["metadata"].get("title"),
                tags=data["metadata"].get("tags", []),
                summary=data["metadata"].get("summary"),
                last_summarized_count=data["metadata"].get("last_summarized_count", 0)
            )
            
            # Create history
            history = EnhancedChatMessageHistory(metadata=metadata)
            
            # Load messages
            for msg_data in data.get("messages", []):
                if msg_data["type"] == "human":
                    history.add_message(HumanMessage(content=msg_data["content"]))
                elif msg_data["type"] == "ai":
                    history.add_message(AIMessage(content=msg_data["content"]))
            
            logger.info(f"Loaded thread {thread_id} with {len(history.messages)} messages")
            return history
            
        except Exception as e:
            logger.warning(f"Could not load thread {thread_id}: {e}")
            return None
    
    def _create_new_thread(self, thread_id: str, user_id: str) -> EnhancedChatMessageHistory:
        """Create a new thread."""
        metadata = ThreadMetadata(
            thread_id=thread_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        history = EnhancedChatMessageHistory(metadata=metadata)
        self._history_cache[thread_id] = history
        
        # Update user mappings
        if user_id not in self._user_threads:
            self._user_threads[user_id] = []
        
        if thread_id not in self._user_threads[user_id]:
            self._user_threads[user_id].append(thread_id)
            
            # Enforce thread limit per user
            if len(self._user_threads[user_id]) > self.max_threads_per_user:
                # Remove oldest thread
                oldest_thread = self._user_threads[user_id].pop(0)
                self._delete_thread(oldest_thread)
                logger.info(f"Removed oldest thread {oldest_thread} for user {user_id}")
        
        self._save_user_mappings()
        logger.info(f"Created new thread {thread_id} for user {user_id}")
        return history
    
    def save_thread(self, thread_id: str) -> None:
        """Save thread to disk."""
        with self._lock:
            if thread_id not in self._history_cache:
                return
            
            history = self._history_cache[thread_id]
            thread_file = self._get_thread_file_path(thread_id)
            
            # Prepare data for saving
            messages_data = []
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    messages_data.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages_data.append({"type": "ai", "content": msg.content})
            
            data = {
                "metadata": {
                    "thread_id": history.metadata.thread_id,
                    "user_id": history.metadata.user_id,
                    "created_at": history.metadata.created_at.isoformat(),
                    "last_accessed": history.metadata.last_accessed.isoformat(),
                    "message_count": history.metadata.message_count,
                    "title": history.metadata.title,
                    "tags": history.metadata.tags,
                    "summary": history.metadata.summary,
                    "last_summarized_count": history.metadata.last_summarized_count
                },
                "messages": messages_data
            }
            
            try:
                with open(thread_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save thread {thread_id}: {e}")
    
    def _delete_thread(self, thread_id: str) -> None:
        """Delete a thread from memory and disk."""
        # Remove from cache
        if thread_id in self._history_cache:
            del self._history_cache[thread_id]
        
        # Remove from disk
        thread_file = self._get_thread_file_path(thread_id)
        if thread_file.exists():
            try:
                thread_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete thread file {thread_id}: {e}")
    
    def clear_thread(self, thread_id: str) -> None:
        """Clear thread messages but keep metadata."""
        with self._lock:
            if thread_id in self._history_cache:
                self._history_cache[thread_id].clear()
                self.save_thread(thread_id)
    
    def get_user_threads(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all threads for a user."""
        with self._lock:
            thread_ids = self._user_threads.get(user_id, [])
            threads = []
            
            for tid in thread_ids:
                if tid in self._history_cache:
                    threads.append(self._history_cache[tid].get_summary())
                else:
                    # Try to load basic info from disk
                    thread_file = self._get_thread_file_path(tid)
                    if thread_file.exists():
                        try:
                            with open(thread_file, "r") as f:
                                data = json.load(f)
                            threads.append({
                                "thread_id": tid,
                                "user_id": user_id,
                                "title": data["metadata"].get("title", f"Thread {tid[:8]}"),
                                "message_count": data["metadata"]["message_count"],
                                "created_at": data["metadata"]["created_at"],
                                "last_accessed": data["metadata"]["last_accessed"],
                                "tags": data["metadata"].get("tags", [])
                            })
                        except Exception:
                            pass
            
            # Sort by last accessed time (most recent first)
            threads.sort(key=lambda x: x.get("last_accessed", ""), reverse=True)
            return threads
    
    def cleanup_old_threads(self, days: int = 30) -> int:
        """Clean up threads older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        with self._lock:
            to_remove = []
            
            for user_id, thread_ids in self._user_threads.items():
                for thread_id in thread_ids[:]:  # Copy list to avoid modification during iteration
                    thread_file = self._get_thread_file_path(thread_id)
                    if thread_file.exists():
                        try:
                            with open(thread_file, "r") as f:
                                data = json.load(f)
                            last_accessed = datetime.fromisoformat(data["metadata"]["last_accessed"])
                            
                            if last_accessed < cutoff_date:
                                to_remove.append((user_id, thread_id))
                        except Exception:
                            # If we can't read the file, consider it for removal
                            to_remove.append((user_id, thread_id))
            
            # Remove old threads
            for user_id, thread_id in to_remove:
                if thread_id in self._user_threads[user_id]:
                    self._user_threads[user_id].remove(thread_id)
                self._delete_thread(thread_id)
                cleaned_count += 1
            
            if cleaned_count > 0:
                self._save_user_mappings()
                logger.info(f"Cleaned up {cleaned_count} old threads")
        
        return cleaned_count


# Global memory manager
_memory_manager: Optional[ThreadSafeMemoryManager] = None

# Global variables for dependencies (like your previous implementation)
global_llm = None
global_retriever = None


# LangGraph State Schema
class RAGState(TypedDict):
    """State for the RAG workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    rewritten_query: str
    context: str
    answer: str
    needs_retrieval: bool  # New field for conditional routing


def get_session_history(thread_id: str, user_id: str = "default") -> BaseChatMessageHistory:
    """Get or create chat history for a session using enhanced memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ThreadSafeMemoryManager()
    
    return _memory_manager.get_or_create_thread(thread_id, user_id)


def save_session_history(thread_id: str) -> None:
    """Persist chat history to disk using enhanced memory manager."""
    global _memory_manager
    if _memory_manager is not None:
        _memory_manager.save_thread(thread_id)


def clear_session_history(thread_id: str) -> None:
    """Clear thread messages using enhanced memory manager."""
    global _memory_manager
    if _memory_manager is not None:
        _memory_manager.clear_thread(thread_id)
        print(f"Cleared thread memory: {thread_id}")


def handle_management_commands(command: str, current_user: str, current_thread: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Handle thread/user management commands.
    
    Returns: (handled, new_user_id, new_thread_id)
    """
    global _memory_manager
    if _memory_manager is None:
        return False, None, None
    
    command = command.lower().strip()
    
    if command == "/threads":
        # List threads for current user
        threads = _memory_manager.get_user_threads(current_user)
        if not threads:
            print(f"No threads found for user '{current_user}'")
        else:
            print(f"\n📋 Threads for user '{current_user}':")
            print("-" * 60)
            for i, thread in enumerate(threads, 1):
                status = "📍 CURRENT" if thread["thread_id"] == current_thread else ""
                print(f"{i:2d}. {thread['title'][:40]:<40} {status}")
                print(f"     ID: {thread['thread_id']}")
                print(f"     Messages: {thread['message_count']:3d} | Last: {thread['last_accessed'][:16]}")
                if thread.get('tags'):
                    print(f"     Tags: {', '.join(thread['tags'])}")
                print()
        return True, None, None
    
    elif command == "/users":
        # List all users with thread counts
        if not _memory_manager._user_threads:
            print("No users found")
        else:
            print("\n👥 All Users:")
            print("-" * 40)
            for user_id, thread_ids in _memory_manager._user_threads.items():
                status = "📍 CURRENT" if user_id == current_user else ""
                print(f"• {user_id:<20} ({len(thread_ids)} threads) {status}")
        return True, None, None
    
    elif command.startswith("/switch"):
        # Switch thread or user
        parts = command.split()
        if len(parts) < 2:
            print("Usage: /switch <thread_id> or /switch user <user_id> [thread_id]")
            return True, None, None
        
        if parts[1] == "user" and len(parts) >= 3:
            # Switch user and optionally thread
            new_user = parts[2]
            new_thread = parts[3] if len(parts) > 3 else str(uuid.uuid4())[:8]
            print(f"Switched to user '{new_user}', thread '{new_thread}'")
            return True, new_user, new_thread
        else:
            # Switch thread only
            new_thread = parts[1]
            print(f"Switched to thread '{new_thread}'")
            return True, None, new_thread
    
    elif command.startswith("/new"):
        # Create new thread
        parts = command.split()
        if len(parts) > 1:
            # Use provided thread name/id
            new_thread = parts[1]
        else:
            # Generate new thread ID
            new_thread = str(uuid.uuid4())[:8]
        print(f"Created new thread '{new_thread}'")
        return True, None, new_thread
    
    elif command.startswith("/title"):
        # Set thread title
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /title <thread title>")
            return True, None, None
        
        title = parts[1]
        if current_thread in _memory_manager._history_cache:
            _memory_manager._history_cache[current_thread].metadata.title = title
            _memory_manager.save_thread(current_thread)
            print(f"Set thread title to: {title}")
        else:
            print("Current thread not found in cache")
        return True, None, None
    
    elif command.startswith("/tag"):
        # Add tag to thread
        parts = command.split()
        if len(parts) < 2:
            print("Usage: /tag <tag_name>")
            return True, None, None
        
        tag = parts[1]
        if current_thread in _memory_manager._history_cache:
            history = _memory_manager._history_cache[current_thread]
            if tag not in history.metadata.tags:
                history.metadata.tags.append(tag)
                _memory_manager.save_thread(current_thread)
                print(f"Added tag: {tag}")
            else:
                print(f"Tag '{tag}' already exists")
        else:
            print("Current thread not found in cache")
        return True, None, None
    
    elif command == "/cleanup":
        # Clean up old threads
        cleaned = _memory_manager.cleanup_old_threads(days=30)
        print(f"Cleaned up {cleaned} old threads (>30 days)")
        return True, None, None
    
    elif command == "/info":
        # Show current session info
        print(f"\n📊 Current Session Info:")
        print(f"User ID: {current_user}")
        print(f"Thread ID: {current_thread}")
        
        if current_thread in _memory_manager._history_cache:
            history = _memory_manager._history_cache[current_thread]
            metadata = history.metadata
            print(f"Thread Title: {metadata.title or 'Untitled'}")
            print(f"Messages: {len(history.messages)}")
            print(f"Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Last Access: {metadata.last_accessed.strftime('%Y-%m-%d %H:%M:%S')}")
            if metadata.tags:
                print(f"Tags: {', '.join(metadata.tags)}")
        
        return True, None, None
    
    return False, None, None


def create_conversation_summary(messages: List[BaseMessage], current_summary: str = None) -> str:
    """Create or update conversation summary using LLM."""
    global global_llm
    
    if not messages:
        return current_summary or ""
    
    # Convert messages to text format for summarization
    conversation_text = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"Employee: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_text.append(f"Assistant: {msg.content}")
    
    conversation = "\n".join(conversation_text)
    
    if current_summary:
        # Update existing summary
        summary_prompt = f"""You are summarizing a conversation between a Bank of England employee and an HR assistant.

Previous conversation summary:
{current_summary}

New conversation to add:
{conversation}

Create an updated summary that:
- Captures key topics discussed
- Maintains important context for future questions
- Includes any employee preferences or specific situations mentioned
- Keeps track of ongoing issues or requests
- Is concise but informative (max 200 words)

Updated summary:"""
    else:
        # Create initial summary
        summary_prompt = f"""You are summarizing a conversation between a Bank of England employee and an HR assistant.

Conversation:
{conversation}

Create a summary that:
- Captures key topics discussed
- Maintains important context for future questions
- Includes any employee preferences or specific situations mentioned
- Keeps track of ongoing issues or requests
- Is concise but informative (max 200 words)

Summary:"""
    
    try:
        if global_llm:
            summary = global_llm.invoke(summary_prompt).content.strip()
            logger.debug(f"Created conversation summary: {len(summary)} chars")
            return summary
        else:
            logger.warning("No LLM available for summarization")
            return current_summary or ""
    except Exception as e:
        logger.error(f"Failed to create summary: {e}")
        return current_summary or ""


def get_contextual_history(history: "EnhancedChatMessageHistory", recent_count: int = 6) -> str:
    """Get conversation context: summary + recent messages."""
    context_parts = []
    
    # Add summary if available
    if history.metadata.summary:
        context_parts.append(f"Previous conversation summary: {history.metadata.summary}")
    
    # Add recent messages
    if len(history.messages) > 0:
        recent_messages = history.messages[-recent_count:] if len(history.messages) > recent_count else history.messages
        
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"Employee: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"Assistant: {msg.content}")
    
    return "\n".join(context_parts) if context_parts else ""


def create_llm_provider(provider: str, model: str) -> ChatOpenAI:
    """Factory function to create LLM provider based on configuration."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0
        )
    
    elif provider == "vllm":
        base_url = os.getenv("VLLM_BASE_URL")
        if not base_url:
            raise ValueError("VLLM_BASE_URL is required for vLLM provider")
        
        api_key = os.getenv("VLLM_API_KEY", "sk-local")  # Default fallback
        vllm_model = os.getenv("VLLM_MODEL", model)  # Use VLLM_MODEL if set, fallback to MODEL
        
        return ChatOpenAI(
            model=vllm_model,
            base_url=base_url,
            api_key=api_key,
            temperature=0
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'vllm'")


def extract_pdf_content(file_path: str) -> str:
    """Extract text from PDF using multiple fallback methods."""
    content = ""
    errors = []
    
    # Method 1: Try PyPDF2/pypdf (fastest, most common)
    for reader_name in ['pypdf2', 'pypdf']:
        if reader_name in PDF_READERS:
            try:
                logger.debug(f"Trying {reader_name} for {file_path}")
                with open(file_path, 'rb') as f:
                    reader = PDF_READERS[reader_name](f)
                    content = "\n".join(page.extract_text() for page in reader.pages)
                if content.strip():
                    logger.info(f"Successfully extracted PDF content using {reader_name}")
                    return content
            except Exception as e:
                errors.append(f"{reader_name}: {e}")
                continue
    
    # Method 2: Try pdfplumber (better for complex layouts, tables)
    if 'pdfplumber' in PDF_READERS:
        try:
            logger.debug(f"Trying pdfplumber for {file_path}")
            with PDF_READERS['pdfplumber'].open(file_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                content = "\n".join(pages)
            if content.strip():
                logger.info("Successfully extracted PDF content using pdfplumber")
                return content
        except Exception as e:
            errors.append(f"pdfplumber: {e}")
    
    # Method 3: Try PyMuPDF (excellent performance and accuracy)
    if 'pymupdf' in PDF_READERS:
        try:
            logger.debug(f"Trying PyMuPDF for {file_path}")
            doc = PDF_READERS['pymupdf'].open(file_path)
            pages = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    pages.append(text)
            content = "\n".join(pages)
            doc.close()
            if content.strip():
                logger.info("Successfully extracted PDF content using PyMuPDF")
                return content
        except Exception as e:
            errors.append(f"pymupdf: {e}")
    
    # If all methods failed, log the errors
    if errors:
        logger.warning(f"All PDF extraction methods failed for {file_path}:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return content


def load_documents(docs_path: str) -> List[Document]:
    """Load documents from the specified directory with enhanced PDF support."""
    documents = []
    
    # Find all supported file types
    patterns = [
        os.path.join(docs_path, "**", "*.md"),
        os.path.join(docs_path, "**", "*.txt"),
        os.path.join(docs_path, "**", "*.docx"),  # Add Word support
    ]
    
    if HAS_PDF:
        patterns.append(os.path.join(docs_path, "**", "*.pdf"))
        logger.info(f"PDF support enabled with: {', '.join(PDF_READERS.keys())}")
    else:
        logger.warning("No PDF libraries found. Install one of: pypdf2, pypdf, pdfplumber, pymupdf")
    
    files = []
    for pattern in patterns:
        files.extend(glob(pattern, recursive=True))
    
    if not files:
        print(f"No documents found in {docs_path}")
        return documents
    
    print(f"Loading {len(files)} files...")
    
    for file_path in tqdm(files):
        try:
            content = ""
            file_size = os.path.getsize(file_path)
            logger.debug(f"Processing {file_path} ({file_size} bytes)")
            
            if file_path.lower().endswith('.pdf'):
                if HAS_PDF:
                    content = extract_pdf_content(file_path)
                    if not content.strip():
                        print(f"Warning: Could not extract text from PDF {file_path}")
                        continue
                else:
                    print(f"Skipping PDF {file_path} (no PDF libraries installed)")
                    continue
            
            elif file_path.lower().endswith('.docx'):
                try:
                    import docx2txt
                    content = docx2txt.process(file_path)
                except ImportError:
                    try:
                        from docx import Document as DocxDocument
                        doc = DocxDocument(file_path)
                        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    except ImportError:
                        print(f"Skipping DOCX {file_path} (install docx2txt or python-docx)")
                        continue
            
            else:
                # Handle text files with better encoding detection
                encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"Warning: Could not decode {file_path} with any encoding")
                    continue
            
            # Validate content
            if content and content.strip():
                # Clean up content
                content = content.strip()
                
                # Add rich metadata
                metadata = {
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": os.path.splitext(file_path)[1].lower(),
                    "file_size": file_size,
                    "char_count": len(content)
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                logger.debug(f"Loaded {file_path}: {len(content)} characters")
            else:
                print(f"Warning: Skipping empty file {file_path}")
                
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            logger.debug(f"Error details for {file_path}: {e}", exc_info=True)
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Print summary by file type
    file_types = {}
    for doc in documents:
        file_type = doc.metadata.get('file_type', 'unknown')
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    if file_types:
        print("📋 Document types:")
        for file_type, count in sorted(file_types.items()):
            print(f"   {file_type}: {count} files")
    
    return documents


def format_docs(docs: List[Document]) -> str:
    """Format documents for context with citations."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:1000]  # First 1000 chars
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{i}] {content}\nSOURCE: {source}")
    
    return "\n\n".join(formatted)


def query_classifier_node(state: RAGState) -> RAGState:
    """Classify if query needs document retrieval or can be answered directly."""
    question = state["question"]
    messages = state["messages"]
    
    logger.info(f"🎯 Classifying query: '{question}'")
    
    # Get conversation context (summary + recent messages) instead of full history
    # For classification, we only need recent context
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    context_text = ""
    
    if len(recent_messages) > 1:
        context_parts = []
        for msg in recent_messages[:-1]:  # Exclude current question
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Employee: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        context_text = "\n".join(context_parts) if context_parts else ""
    
    # Classification prompt
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier for an internal company knowledge system.

Analyze the user's question and determine if it needs document retrieval from company documents.

**ALWAYS RETRIEVE** for ANY company-specific information including:
- Benefits, leave policies, vacation days, PTO
- Employee handbook topics and HR policies  
- Company procedures, processes, workflows
- Technical guidelines, standards, protocols
- Internal forms, applications, requirements
- Salary, compensation, performance reviews
- Training, onboarding, compliance
- Company dates, deadlines, schedules
- Office policies, equipment, facilities
- ANY question about "How many...", "What dates...", "How do I..." related to work
- ANY question that could have a company-specific answer

**ONLY DIRECT ANSWER** for clearly general knowledge:
- Basic math calculations (2+2, percentages)
- Common facts (capitals of countries, historical dates)
- General explanations of concepts (not company-specific)
- Simple greetings like "hello" or "how are you"

When in doubt, choose RETRIEVE. It's better to search documents than miss company-specific information.

Respond with ONLY:
- "RETRIEVE" if the answer might be in company documents
- "DIRECT" if it's clearly general knowledge unrelated to the company"""),
        ("human", f"Recent context:\n{context_text}\n\nCurrent question: {question}")
    ])
    
    classifier_chain = classify_prompt | global_llm | (lambda x: x.content.strip())
    decision = classifier_chain.invoke({})
    
    needs_retrieval = decision.upper() == "RETRIEVE"
    logger.info(f"📋 Classification: {decision} → needs_retrieval={needs_retrieval}")
    
    return {"needs_retrieval": needs_retrieval}


def rewrite_query_node(state: RAGState) -> RAGState:
    """Node to rewrite the query using conversation context."""
    question = state["question"]
    messages = state["messages"]
    needs_retrieval = state["needs_retrieval"]
    
    # Only rewrite if we need retrieval
    if not needs_retrieval:
        logger.info(f"🔄 Query: '{question}' (direct answer - no rewrite needed)")
        return {"rewritten_query": question}
    
    # Get recent conversation context for query rewriting
    recent_messages = messages[-8:] if len(messages) > 8 else messages
    context_text = ""
    
    if len(recent_messages) > 1:
        context_parts = []
        for msg in recent_messages[:-1]:  # Exclude current question
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Employee: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        context_text = "\n".join(context_parts) if context_parts else ""
    
    # If no context, use original question
    if not context_text:
        rewritten_query = question
        logger.info(f"🔄 Query: '{question}' (no context)")
    else:
        # Context-aware query rewriter
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You rewrite the user's latest question into a standalone query using the conversation context. Return ONLY the rewritten query."),
            ("human", f"Recent conversation:\n{context_text}\n\nLatest question: {question}\n\nRewritten standalone query:")
        ])
        
        rewriter_chain = rewrite_prompt | global_llm | (lambda x: x.content)
        rewritten_query = rewriter_chain.invoke({})
        logger.info(f"🔄 Query: '{question}' → '{rewritten_query}'")
    
    return {"rewritten_query": rewritten_query}


def retrieve_node(state: RAGState) -> RAGState:
    """Node to retrieve relevant documents (only if needed)."""
    needs_retrieval = state["needs_retrieval"]
    rewritten_query = state["rewritten_query"]
    
    # Skip retrieval if not needed
    if not needs_retrieval:
        logger.info("🔍 Skipping retrieval (direct answer)")
        return {"context": ""}
    
    logger.info(f"🔍 Retrieving docs for: '{rewritten_query}'")
    
    # Retrieve documents
    docs = global_retriever.get_relevant_documents(rewritten_query)
    
    # Log sources found
    sources = [doc.metadata.get("source", "Unknown").split("/")[-1] for doc in docs]
    logger.info(f"📚 Found {len(docs)} docs: {', '.join(sources)}")
    
    context = format_docs(docs)
    return {"context": context}


def generate_node(state: RAGState) -> RAGState:
    """Node to generate the final answer."""
    question = state["question"]
    context = state["context"]
    messages = state["messages"]
    needs_retrieval = state["needs_retrieval"]
    
    # Get contextual history (summary + recent messages) instead of full history
    # Find the session history to get the contextual information
    session_history = None
    thread_id = None
    
    # Try to extract thread info from messages or use a simplified context approach
    recent_messages = messages[-10:] if len(messages) > 10 else messages
    conversation_context = ""
    
    if len(recent_messages) > 1:
        context_parts = []
        for msg in recent_messages[:-1]:  # Exclude current question
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Employee: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        conversation_context = "\n".join(context_parts) if context_parts else ""
    
    if needs_retrieval and context:
        # Retrieval-based answer with context
        logger.info("🤖 Generating retrieval-based answer")
        prompt_content = f"""You are a helpful Bank of England HR assistant. Answer the employee's question using the information available to you. Be natural and conversational. Do not mention 'context', 'documents', or 'provided information' - just answer as if you naturally know this information. If you don't have the specific information needed, politely say you don't have that information available.

Available information:
{context}

{f"Recent conversation context:{conversation_context}" if conversation_context else ""}

Employee question: {question}"""

        answer_prompt = ChatPromptTemplate.from_messages([
            ("human", prompt_content)
        ])
        
        answer = answer_prompt | global_llm | (lambda x: x.content)
        response = answer.invoke({})
        
    else:
        # Direct answer without retrieval
        logger.info("🤖 Generating direct answer (general knowledge)")
        prompt_content = f"""You are a friendly Bank of England assistant. Answer the employee's question naturally and conversationally. For greetings and casual conversation, be warm and professional. For general knowledge questions, provide helpful answers while keeping in mind you're assisting a Bank of England employee.

{f"Recent conversation context:{conversation_context}" if conversation_context else ""}

Employee question: {question}"""

        direct_prompt = ChatPromptTemplate.from_messages([
            ("human", prompt_content)
        ])
        
        answer = direct_prompt | global_llm | (lambda x: x.content)
        response = answer.invoke({})
    
    logger.info(f"💬 Generated answer ({len(response)} chars)")
    
    return {"answer": response}


def build_rag_graph(llm: ChatOpenAI, retriever, k: int):
    """Build the LangGraph-based RAG workflow with conditional routing."""
    global global_llm, global_retriever
    
    # Set global dependencies (like your previous implementation)
    global_llm = llm
    global_retriever = retriever
    
    # Create the state graph
    graph_builder = StateGraph(RAGState)
    
    # Add nodes (clean approach like your previous implementation)
    graph_builder.add_node(query_classifier_node)
    graph_builder.add_node(rewrite_query_node)
    graph_builder.add_node(retrieve_node)
    graph_builder.add_node(generate_node)
    
    # Define the flow with conditional routing
    graph_builder.set_entry_point("query_classifier_node")
    graph_builder.add_edge("query_classifier_node", "rewrite_query_node")
    graph_builder.add_edge("rewrite_query_node", "retrieve_node")
    graph_builder.add_edge("retrieve_node", "generate_node")
    graph_builder.add_edge("generate_node", END)
    
    # Compile the graph with enhanced memory checkpointing
    # Use MemorySaver for LangGraph's internal state management
    # Our enhanced memory system handles the chat history persistence
    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    logger.info("🏗️ Enhanced RAG workflow ready (with conditional routing & enhanced memory)")
    return graph


def main():
    """Main chat loop."""
    parser = argparse.ArgumentParser(description="Interactive RAG chatbot with enhanced memory management")
    parser.add_argument("--docs", default="docs", help="Documents directory")
    parser.add_argument("--thread", default="default", help="Thread ID for persistent memory")
    parser.add_argument("--user", default="default", help="User ID for session management")
    parser.add_argument("--k", type=int, default=2, help="Number of documents to retrieve")
    parser.add_argument("--max-messages", type=int, default=100, help="Maximum messages per thread")
    parser.add_argument("--max-threads", type=int, default=10, help="Maximum threads per user")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging to see workflow state transfers")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Initialize global memory manager with user settings
    global _memory_manager
    _memory_manager = ThreadSafeMemoryManager(max_threads_per_user=args.max_threads)
    
    # Configure logging based on arguments
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level)
    
    # Update logger level
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    
    # Load environment
    load_dotenv()
    
    # Get configuration
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("MODEL", "gpt-4o-mini")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    
    print(f"Using provider: {provider}, model: {model}")
    print(f"Memory: max {args.max_messages} messages/thread, {args.max_threads} threads/user")
    
    try:
        # Create LLM
        llm = create_llm_provider(provider, model)
        
        # Create embeddings (always OpenAI for now)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        
        embeddings = OpenAIEmbeddings(
            model=embed_model,
            api_key=openai_api_key
        )
        
        # Load and index documents
        documents = load_documents(args.docs)
        if not documents:
            print("No documents found. Please add files to the docs directory.")
            sys.exit(1)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        
        print("Splitting documents...")
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        
        # Create vector store
        print("Building FAISS index...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
        
        # Build RAG graph
        rag_graph = build_rag_graph(llm, retriever, args.k)
        
        # Initialize session variables
        current_user = args.user
        current_thread = args.thread
        
        print(f"\n✅ Ready! User: {current_user}, Thread: {current_thread}")
        print("💬 Enhanced Commands:")
        print("  Basic: /exit, /quit, /reset, /workflow")
        print("  Memory: /threads, /users, /switch, /new, /info")
        print("  Thread: /title <name>, /tag <name>, /cleanup")
        print("  Usage: /switch <thread_id> | /switch user <user_id> [thread_id]")
        if args.verbose:
            print("💡 Verbose logging enabled - you'll see workflow state transfers")
        
        # Set max messages for new threads
        if current_thread in _memory_manager._history_cache:
            _memory_manager._history_cache[current_thread].max_messages = args.max_messages
        
        # Interactive loop
        while True:
            try:
                user_input = input(f"{current_user}> ").strip()
                
                if user_input.lower() in ["/exit", "/quit"]:
                    save_session_history(current_thread)
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "/reset":
                    clear_session_history(current_thread)
                    print("Thread memory cleared.")
                    continue
                
                if user_input.lower() == "/workflow":
                    visualize_graph_structure()
                    continue
                
                # Handle management commands
                handled, new_user, new_thread = handle_management_commands(user_input, current_user, current_thread)
                if handled:
                    # Save current thread before switching
                    save_session_history(current_thread)
                    
                    # Update session variables if changed
                    if new_user:
                        current_user = new_user
                    if new_thread:
                        current_thread = new_thread
                        # Set max messages for new thread
                        history = get_session_history(current_thread, current_user)
                        if hasattr(history, 'max_messages'):
                            history.max_messages = args.max_messages
                    continue
                
                if not user_input:
                    continue
                
                # Process query
                start_time = time.time()
                
                if args.verbose:
                    logger.info(f"🚀 Processing: '{user_input}'")
                
                # Get current session history with user context
                session_history = get_session_history(current_thread, current_user)
                
                # Create initial state with user message
                user_message = HumanMessage(content=user_input)
                initial_state = {
                    "messages": session_history.messages + [user_message],
                    "question": user_input,
                    "rewritten_query": "",
                    "context": "",
                    "answer": "",
                    "needs_retrieval": False  # Will be determined by classifier
                }
                
                # Run the graph
                result = rag_graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": current_thread}}
                )
                
                # Get the answer
                answer = result["answer"]
                elapsed = time.time() - start_time
                
                if args.verbose:
                    logger.info(f"✅ Complete ({elapsed:.2f}s)\n")
                
                print(f"bot> {answer}")
                print(f"(time: {elapsed:.2f}s)")
                
                # Add messages to session history
                session_history.add_message(user_message)
                session_history.add_message(AIMessage(content=answer))
                
                # Save history after each turn
                save_session_history(current_thread)
                
            except KeyboardInterrupt:
                save_session_history(current_thread)
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Startup error: {e}")
        sys.exit(1)


# Optional: Example of how to use the vLLM adapter (commented out)
# from src.adapters.vllm_client_adapter import make_vllm_runnable, SimpleVLLMClient
# 
# def create_vllm_with_adapter():
#     """Alternative vLLM setup using custom adapter."""
#     client = SimpleVLLMClient(
#         base_url=os.getenv("VLLM_BASE_URL"),
#         api_key=os.getenv("VLLM_API_KEY")
#     )
#     return make_vllm_runnable(client)


def visualize_graph_structure():
    """Optional function to visualize the enhanced LangGraph structure."""
    print("\n🔄 Enhanced RAG Workflow Structure (with Conditional Routing):")
    print("┌─────────────────┐")
    print("│   User Query    │")
    print("└─────────┬───────┘")
    print("          │")
    print("   ┌──────▼──────┐")
    print("   │ Classify    │ ← Decides: company docs vs general knowledge")
    print("   │ Query Node  │")
    print("   └──────┬──────┘")
    print("          │")
    print("   ┌──────▼──────┐")
    print("   │ Rewrite     │ ← History-aware rewrite (if needed)")
    print("   │ Query Node  │")
    print("   └──────┬──────┘")
    print("          │")
    print("   ┌──────▼──────┐")
    print("   │ Retrieve    │ ← Search FAISS (if company docs needed)")
    print("   │ Node        │   Otherwise skip")
    print("   └──────┬──────┘")
    print("          │")
    print("   ┌──────▼──────┐")
    print("   │ Generate    │ ← Context-based OR direct answer")
    print("   │ Node        │")
    print("   └──────┬──────┘")
    print("          │")
    print("   ┌──────▼──────┐")
    print("   │   Answer    │")
    print("   └─────────────┘")
    print("\n🎯 Smart routing: Only retrieves when needed!")
    print("💾 Enhanced Memory System:")
    print("   • Thread-safe user/thread management")
    print("   • Auto-cleanup of old conversations") 
    print("   • Configurable memory limits")
    print("   • Rich metadata & tagging support")
    print("   • LangGraph + Enhanced Memory integration")


if __name__ == "__main__":
    main()

