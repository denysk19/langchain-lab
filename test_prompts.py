#!/usr/bin/env python3
"""Test script for the prompt configuration system."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag.prompt_manager import PromptManager

def test_prompt_manager():
    """Test the PromptManager functionality."""
    print("🧪 Testing PromptManager...")
    
    # Test default configuration
    pm = PromptManager()
    print("✅ PromptManager initialized")
    
    # Test classification prompts
    context = "Employee: Hello\nAssistant: Hi there!"
    question = "How many vacation days do I get?"
    
    system, human = pm.get_classification_prompts(context, question)
    print(f"📋 Classification system prompt (first 100 chars): {system[:100]}...")
    print(f"📋 Classification human prompt: {human}")
    
    # Test rewrite prompts
    system, human = pm.get_rewrite_prompts(context, question)
    print(f"🔄 Rewrite system prompt: {system}")
    
    # Test generation prompts - retrieval mode
    system, human = pm.get_generation_prompts(
        mode="retrieval_based",
        context="Company policy states...",
        conversation_context="Recent conversation: ...",
        question=question
    )
    print(f"🤖 Generation system prompt (retrieval): {system[:100]}...")
    
    # Test generation prompts - direct mode
    system, human = pm.get_generation_prompts(
        mode="direct_answer",
        conversation_context="Recent conversation: ...",
        question="What is 2+2?"
    )
    print(f"🤖 Generation system prompt (direct): {system[:100]}...")
    
    # Test summarization
    summary_prompt = pm.get_summarization_prompt("Employee: Hi\nAssistant: Hello!")
    print(f"📝 Summarization prompt (first 100 chars): {summary_prompt[:100]}...")
    
    # Test available variants
    variants = pm.get_available_variants()
    print(f"🎯 Available variants: {variants}")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_prompt_manager()
