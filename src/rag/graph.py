"""RAG workflow graph construction."""

import logging
from langgraph.graph import StateGraph, END
from .state import RAGState
from .nodes import RAGNodes

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """RAG workflow builder and manager."""
    
    def __init__(self, llm, retriever, k: int = 2):
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.nodes = RAGNodes(llm, retriever)
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph-based RAG workflow with conditional routing."""
        logger.info("🏗️ Building RAG workflow graph...")
        
        # Create the state graph
        graph_builder = StateGraph(RAGState)
        
        # Add nodes
        graph_builder.add_node("query_classifier", self.nodes.query_classifier_node)
        graph_builder.add_node("rewrite_query", self.nodes.rewrite_query_node)
        graph_builder.add_node("retrieve", self.nodes.retrieve_node)
        graph_builder.add_node("generate", self.nodes.generate_node)
        
        # Define the flow with conditional routing
        graph_builder.set_entry_point("query_classifier")
        graph_builder.add_edge("query_classifier", "rewrite_query")
        graph_builder.add_edge("rewrite_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        
        # Compile the graph (no checkpointer needed for simple linear workflow)
        graph = graph_builder.compile()
        
        logger.info("🏗️ RAG workflow ready (with conditional routing)")
        return graph
    
    def invoke(self, initial_state: dict) -> dict:
        """Execute the RAG workflow."""
        return self.graph.invoke(initial_state)
    
    def stream(self, initial_state: dict):
        """Stream the RAG workflow execution."""
        return self.graph.stream(initial_state)


def create_rag_workflow(llm, retriever, k: int = 2) -> RAGWorkflow:
    """Factory function to create a RAG workflow."""
    return RAGWorkflow(llm, retriever, k)
