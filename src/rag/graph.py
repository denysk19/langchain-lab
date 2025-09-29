"""RAG workflow graph construction."""

import logging
from langgraph.graph import StateGraph, END
from .state import RAGState
from .nodes import RAGNodes
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """RAG workflow builder and manager."""
    
    def __init__(self, llm, retriever, k: int = 2, prompt_manager: PromptManager = None):
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.prompt_manager = prompt_manager or PromptManager()
        self.nodes = RAGNodes(llm, retriever, self.prompt_manager)
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
        
        logger.info("🏗️ RAG workflow ready (with configurable prompts)")
        return graph
    
    def invoke(self, initial_state: dict) -> dict:
        """Execute the RAG workflow."""
        return self.graph.invoke(initial_state)
    
    def stream(self, initial_state: dict):
        """Stream the RAG workflow execution."""
        return self.graph.stream(initial_state)
    
    def reload_prompts(self):
        """Reload prompts from configuration."""
        self.prompt_manager.reload_prompts()
        logger.info("Prompts reloaded in RAG workflow")


def create_rag_workflow(llm, retriever, k: int = 2, 
                       prompt_config: str = None, prompt_variant: str = "default") -> RAGWorkflow:
    """Factory function to create a RAG workflow with configurable prompts."""
    prompt_manager = PromptManager(config_path=prompt_config, variant=prompt_variant)
    return RAGWorkflow(llm, retriever, k, prompt_manager)
