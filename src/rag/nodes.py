"""RAG workflow nodes."""

import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from .state import RAGState
from .utils import format_docs, extract_conversation_context
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class RAGNodes:
    """Container for RAG workflow nodes."""
    
    def __init__(self, llm, retriever, prompt_manager: PromptManager = None):
        self.llm = llm
        self.retriever = retriever
        self.prompt_manager = prompt_manager or PromptManager()
    
    def query_classifier_node(self, state: RAGState) -> Dict[str, Any]:
        """Classify if query needs document retrieval or can be answered directly."""
        question = state["question"]
        messages = state["messages"]
        
        logger.debug("=" * 70)
        logger.debug("🎯 QUERY CLASSIFIER NODE")
        logger.debug("=" * 70)
        logger.info(f"🎯 Classifying query: '{question}'")
        
        # Get conversation context
        context_text = extract_conversation_context(messages, max_messages=6)
        
        # Get prompts from configuration
        system_message, human_message = self.prompt_manager.get_classification_prompts(
            context=context_text, 
            question=question
        )
        
        full_prompt = f"System: {system_message}\n\nHuman: {human_message}"
        
        logger.debug(f"Context length: {len(context_text)} chars")
        logger.debug(f"Full prompt length: {len(full_prompt)} chars")
        logger.debug("Full message being sent to vLLM:")
        logger.debug("-" * 50)
        logger.debug(full_prompt)
        logger.debug("-" * 50)
        
        classify_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
        
        classifier_chain = classify_prompt | self.llm | (lambda x: x.content.strip())
        decision = classifier_chain.invoke({})
        
        needs_retrieval = decision.upper() == "RETRIEVE"
        logger.info(f"📋 Classification: {decision} → needs_retrieval={needs_retrieval}")
        logger.debug("=" * 70)
        
        return {"needs_retrieval": needs_retrieval}
    
    def rewrite_query_node(self, state: RAGState) -> Dict[str, Any]:
        """Node to rewrite the query using conversation context."""
        question = state["question"]
        messages = state["messages"]
        needs_retrieval = state["needs_retrieval"]
        
        logger.debug("=" * 70)
        logger.debug("🔄 QUERY REWRITER NODE")
        logger.debug("=" * 70)
        logger.debug(f"Input question: '{question}'")
        logger.debug(f"Needs retrieval: {needs_retrieval}")
        
        # Only rewrite if we need retrieval
        if not needs_retrieval:
            logger.info("🔄 No rewrite needed (direct answer)")
            logger.debug("=" * 70)
            return {"rewritten_query": question}
        
        # Get conversation context
        context_text = extract_conversation_context(messages, max_messages=8)
        
        # If no context, use original question
        if not context_text:
            rewritten_query = question
            logger.info("🔄 No conversation context available")
            logger.debug("=" * 70)
        else:
            # Get prompts from configuration
            system_message, human_message = self.prompt_manager.get_rewrite_prompts(
                context=context_text,
                question=question
            )
            
            full_prompt = f"System: {system_message}\n\nHuman: {human_message}"
            
            logger.debug(f"Context length: {len(context_text)} chars")
            logger.debug(f"Full prompt length: {len(full_prompt)} chars")
            logger.debug("Full message being sent to vLLM:")
            logger.debug("-" * 50)
            logger.debug(full_prompt)
            logger.debug("-" * 50)
            
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", human_message)
            ])
            
            rewriter_chain = rewrite_prompt | self.llm | (lambda x: x.content)
            rewritten_query = rewriter_chain.invoke({})
            logger.info(f"🔄 Rewrite: '{question}' → '{rewritten_query}'")
            logger.debug("=" * 70)
        
        return {"rewritten_query": rewritten_query}
    
    def retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Node to retrieve relevant documents (only if needed)."""
        needs_retrieval = state["needs_retrieval"]
        rewritten_query = state["rewritten_query"]
        
        logger.debug("=" * 70)
        logger.debug("🔍 DOCUMENT RETRIEVAL NODE")
        logger.debug("=" * 70)
        logger.debug(f"Needs retrieval: {needs_retrieval}")
        logger.debug(f"Query for retrieval: '{rewritten_query}'")
        
        # Skip retrieval if not needed
        if not needs_retrieval:
            logger.info("🔍 Skipping retrieval (direct answer)")
            logger.debug("=" * 70)
            return {"context": ""}
        
        # Retrieve documents
        docs = self.retriever.get_relevant_documents(rewritten_query)
        
        # Log sources found
        sources = [doc.metadata.get("source", "Unknown").split("/")[-1] for doc in docs]
        logger.info(f"🔍 Retrieved {len(docs)} documents: {', '.join(sources)}")
        
        context = format_docs(docs)
        logger.debug(f"Formatted context length: {len(context)} chars")
        logger.debug("Context preview (first 200 chars):")
        logger.debug("-" * 50)
        logger.debug(context[:200] + "..." if len(context) > 200 else context)
        logger.debug("-" * 50)
        logger.debug("=" * 70)
        
        return {"context": context}
    
    def generate_node(self, state: RAGState) -> Dict[str, Any]:
        """Node to generate the final answer."""
        question = state["question"]
        context = state["context"]
        messages = state["messages"]
        needs_retrieval = state["needs_retrieval"]
        
        logger.debug("=" * 70)
        logger.debug("🤖 ANSWER GENERATION NODE")
        logger.debug("=" * 70)
        logger.debug(f"Input question: '{question}'")
        logger.debug(f"Needs retrieval: {needs_retrieval}")
        logger.debug(f"Retrieved context length: {len(context)} chars")
        
        # Get conversation context
        conversation_context = extract_conversation_context(messages, max_messages=10)
        logger.debug(f"Conversation context length: {len(conversation_context)} chars")
        
        if needs_retrieval and context:
            # Retrieval-based answer with context
            logger.info("🤖 Generating retrieval-based answer")
            system_message, human_message = self.prompt_manager.get_generation_prompts(
                mode="retrieval_based",
                context=context,
                conversation_context=conversation_context,
                question=question
            )
        else:
            # Direct answer without retrieval
            logger.info("🤖 Generating direct answer (general knowledge)")
            system_message, human_message = self.prompt_manager.get_generation_prompts(
                mode="direct_answer",
                conversation_context=conversation_context,
                question=question
            )
        
        logger.debug(f"Full prompt length: {len(human_message)} chars")
        logger.debug("Full message being sent to vLLM:")
        logger.debug("-" * 50)
        logger.debug(f"System: {system_message}")
        logger.debug(f"Human: {human_message}")
        logger.debug("-" * 50)
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
        
        answer = answer_prompt | self.llm | (lambda x: x.content)
        response = answer.invoke({})
        
        logger.info(f"💬 Generated answer: {response}")
        logger.debug(f"Generated answer length: {len(response)} chars")
        logger.debug("=" * 70)
        
        return {"answer": response}
