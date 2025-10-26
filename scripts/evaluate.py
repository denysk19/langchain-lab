#!/usr/bin/env python3
"""
RAG Evaluation Framework - Inference Stage

Runs the RAG system through golden datasets and stores comprehensive results
with metadata tracking for experiment comparison and analysis.

Debug Mode:
-----------
When enabled with --log-debug-information flag, additional debug columns are added to CSV:
- debug_classification_latency: Time spent in query classification (seconds)
- debug_rewrite_latency: Time spent in query rewriting (seconds)
- debug_retrieval_latency: Time spent in document retrieval (seconds)
- debug_generation_latency: Time spent in answer generation (seconds)
- debug_retrieved_sources: Source filenames of retrieved documents
- debug_retrieved_doc_ids: Document IDs of retrieved documents
- debug_chunk_scores: Similarity scores of retrieved chunks
- debug_original_query_length: Character length of original query
- debug_rewritten_query_length: Character length of rewritten query
- debug_context_length: Character length of retrieved context
- debug_answer_length: Character length of generated answer
- debug_workflow_state: Step-by-step workflow execution trace
- debug_classification_prompt: Full prompt sent to LLM for classification
- debug_rewrite_prompt: Full prompt sent to LLM for query rewriting
- debug_generation_prompt: Full prompt sent to LLM for answer generation
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# Import RAG workflow
from rag_workflow import create_rag_workflow
from rag_workflow.prompt_manager import PromptManager
from rag_workflow.utils import extract_conversation_context

# Import ingestion module
try:
    from ingestion import ingest_pdf, MemoryStore, DocumentCtx
    from ingestion.config import get_config
    HAS_INGESTION = True
except ImportError:
    HAS_INGESTION = False
    print("ERROR: Ingestion module not found!")
    sys.exit(1)

# Import memory retriever adapter
from src.adapters import MemoryRetrieverAdapter

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Show configuration and progress info
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAG_EVAL")
logger.setLevel(logging.INFO)


# Cost estimation data (per 1K tokens)
COST_PER_1K_TOKENS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.0001,
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost for a given model and token counts.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    if model not in COST_PER_1K_TOKENS:
        return 0.0
    
    rates = COST_PER_1K_TOKENS[model]
    if isinstance(rates, dict):
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000
    return (input_tokens + output_tokens) * rates / 1000


def format_ground_truth_context(context: str) -> str:
    """
    Format ground truth context for better readability in CSV.
    Adds a header to match the style of retrieved_context.
    
    Args:
        context: Raw context text from golden dataset
        
    Returns:
        Formatted context with header
    """
    if not context or not context.strip():
        return ""
    
    return f"[Ground Truth Context]\n{context}\nSOURCE: Golden Dataset"


def extract_debug_information(workflow, initial_state: dict, enable_debug: bool, 
                             prompt_manager: Optional[PromptManager] = None) -> Dict[str, Any]:
    """
    Extract debug information from RAG workflow execution.
    
    Args:
        workflow: RAG workflow instance
        initial_state: Initial state for workflow
        enable_debug: Whether to collect detailed debug info
        prompt_manager: PromptManager instance for reconstructing prompts
        
    Returns:
        Dictionary containing debug information and final result
    """
    if not enable_debug:
        # Fast path - just run the workflow normally
        result = workflow.invoke(initial_state)
        return {
            "result": result,
            "debug": {}
        }
    
    # Debug mode - collect intermediate states
    debug_info = {
        "classification_latency": 0,
        "rewrite_latency": 0,
        "retrieval_latency": 0,
        "generation_latency": 0,
        "retrieved_sources": "",
        "retrieved_doc_ids": "",
        "chunk_scores": "",
        "original_query_length": len(initial_state["question"]),
        "rewritten_query_length": 0,
        "context_length": 0,
        "answer_length": 0,
        "workflow_state": "",
        "classification_prompt": "",
        "rewrite_prompt": "",
        "generation_prompt": ""
    }
    
    # Stream through workflow to capture intermediate timing
    step_outputs = []
    step_times = []
    
    try:
        # Start with a copy of the initial state to accumulate updates
        accumulated_state = initial_state.copy()
        
        start_time = time.time()
        for step_output in workflow.stream(initial_state):
            step_end_time = time.time()
            step_outputs.append(step_output)
            step_times.append(step_end_time - start_time)
            
            # Accumulate state updates from this node
            # Each step_output is a dict like {"node_name": {updates}}
            for node_name, updates in step_output.items():
                accumulated_state.update(updates)
            
            start_time = step_end_time
        
        # Parse timing for each node
        if len(step_times) >= 1:
            debug_info["classification_latency"] = step_times[0]
        if len(step_times) >= 2:
            debug_info["rewrite_latency"] = step_times[1]
        if len(step_times) >= 3:
            debug_info["retrieval_latency"] = step_times[2]
        if len(step_times) >= 4:
            debug_info["generation_latency"] = step_times[3]
        
        # Use the accumulated state as the final result
        result = accumulated_state
        
        # Extract additional debug info from result
        if "rewritten_query" in result:
            debug_info["rewritten_query_length"] = len(result["rewritten_query"])
        if "context" in result:
            debug_info["context_length"] = len(result["context"])
        if "answer" in result:
            debug_info["answer_length"] = len(result["answer"])
        
        # Create workflow state summary
        workflow_summary = []
        for i, step_output in enumerate(step_outputs):
            node_name = list(step_output.keys())[0] if step_output else f"step_{i}"
            workflow_summary.append(f"Step {i+1}: {node_name}")
        debug_info["workflow_state"] = " → ".join(workflow_summary)
        
        # Reconstruct prompts if prompt_manager is available
        if prompt_manager:
            question = result.get("question", "")
            messages = result.get("messages", [])
            context = result.get("context", "")
            needs_retrieval = result.get("needs_retrieval", False)
            
            # Extract conversation context
            conversation_context = extract_conversation_context(messages, max_messages=6)
            
            # 1. Classification prompt
            system_msg, human_msg = prompt_manager.get_classification_prompts(
                context=conversation_context,
                question=question
            )
            debug_info["classification_prompt"] = f"SYSTEM:\n{system_msg}\n\nHUMAN:\n{human_msg}"
            
            # 2. Rewrite prompt (only if retrieval was needed)
            if needs_retrieval:
                conversation_context_rewrite = extract_conversation_context(messages, max_messages=8)
                system_msg, human_msg = prompt_manager.get_rewrite_prompts(
                    context=conversation_context_rewrite,
                    question=question
                )
                debug_info["rewrite_prompt"] = f"SYSTEM:\n{system_msg}\n\nHUMAN:\n{human_msg}"
            else:
                debug_info["rewrite_prompt"] = "N/A (no retrieval needed)"
            
            # 3. Generation prompt
            conversation_context_gen = extract_conversation_context(messages, max_messages=10)
            if needs_retrieval and context:
                system_msg, human_msg = prompt_manager.get_generation_prompts(
                    mode="retrieval_based",
                    context=context,
                    conversation_context=conversation_context_gen,
                    question=question
                )
            else:
                system_msg, human_msg = prompt_manager.get_generation_prompts(
                    mode="direct_answer",
                    conversation_context=conversation_context_gen,
                    question=question
                )
            debug_info["generation_prompt"] = f"SYSTEM:\n{system_msg}\n\nHUMAN:\n{human_msg}"
        
    except Exception as e:
        logger.warning(f"Debug info extraction failed: {e}")
        # Fallback to normal execution
        result = workflow.invoke(initial_state)
    
    return {
        "result": result,
        "debug": debug_info
    }


def load_golden_dataset(path: str) -> List[Dict]:
    """
    Load golden dataset from a file or directory.
    
    Args:
        path: Path to JSON file or directory containing JSON files
        
    Returns:
        List of question dictionaries
    """
    if os.path.isfile(path):
        # Single JSON file
        with open(path, 'r') as f:
            return json.load(f)
    else:
        # Directory of JSON files
        data = []
        json_files = sorted(glob(os.path.join(path, "*.json")))
        if not json_files:
            raise ValueError(f"No JSON files found in {path}")
        
        for file in json_files:
            with open(file, 'r') as f:
                data.extend(json.load(f))
        return data


def generate_experiment_name(llm_model: str, custom_name: Optional[str] = None) -> str:
    """
    Generate experiment folder name.
    
    Args:
        llm_model: LLM model name
        custom_name: Optional custom experiment name
        
    Returns:
        Experiment folder name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if custom_name:
        return f"{timestamp}__{custom_name}"
    
    model_short = llm_model.split("/")[-1].replace(".", "_")
    return f"{timestamp}__{model_short}"


def get_git_commit(path: str = ".") -> str:
    """
    Get current git commit hash.
    
    Args:
        path: Path to git repository
        
    Returns:
        Commit hash or "unknown"
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:7]
    except Exception:
        pass
    return "unknown"


def create_llm_provider(provider: str, model: str) -> ChatOpenAI:
    """
    Factory function to create LLM provider.
    
    Args:
        provider: Provider name ('openai' or 'vllm')
        model: Model name
        
    Returns:
        ChatOpenAI instance
    """
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
        
        api_key = os.getenv("VLLM_API_KEY", "sk-local")
        vllm_model = os.getenv("VLLM_MODEL", model)
        
        return ChatOpenAI(
            model=vllm_model,
            base_url=base_url,
            api_key=api_key,
            temperature=0
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def load_and_ingest_documents(docs_path: str, memory_store: MemoryStore, 
                              ctx: DocumentCtx, chunk_size: Optional[int] = None, 
                              overlap: Optional[int] = None) -> int:
    """
    Load documents from directory and ingest into MemoryStore.
    
    Args:
        docs_path: Path to documents directory
        memory_store: MemoryStore instance
        ctx: Document context
        chunk_size: Optional chunk size override
        overlap: Optional overlap override
        
    Returns:
        Total number of chunks ingested
    """
    pdf_files = glob(os.path.join(docs_path, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        logger.warning(f"No PDF documents found in {docs_path}")
        return 0
    
    logger.info(f"Loading {len(pdf_files)} PDF files...")
    
    total_chunks = 0
    
    for file_path in tqdm(pdf_files, desc="Ingesting documents"):
        try:
            file_name = os.path.basename(file_path)
            doc_id = file_name.replace('.pdf', '').replace(' ', '_')
            
            doc_ctx = DocumentCtx(
                tenant_id=ctx.tenant_id,
                owner_user_id=ctx.owner_user_id,
                document_id=doc_id,
                visibility=ctx.visibility,
                embedding_version=ctx.embedding_version
            )
            
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            
            ingest_kwargs = {
                'ctx': doc_ctx,
                'filename': file_name,
                'raw_pdf_bytes': pdf_bytes,
                'sink': memory_store,
            }
            if chunk_size is not None:
                ingest_kwargs['chunk_size'] = chunk_size
            if overlap is not None:
                ingest_kwargs['overlap'] = overlap
            
            result = ingest_pdf(**ingest_kwargs)
            total_chunks += result['chunks_saved']
            
        except Exception as e:
            logger.error(f"Could not ingest {file_path}: {e}")
    
    logger.info(f"Ingested {total_chunks} total chunks from {len(pdf_files)} documents")
    return total_chunks


def run_evaluation(args):
    """Main evaluation runner."""
    
    # Load environment
    load_dotenv()
    
    # Get configurations
    ingestion_config = get_config()
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Get the appropriate model based on provider
    if provider == "vllm":
        model = os.getenv("VLLM_MODEL", os.getenv("MODEL", "gpt-4o-mini"))
    else:
        model = os.getenv("MODEL", "gpt-4o-mini")
    
    logger.info("=" * 80)
    logger.info("RAG EVALUATION FRAMEWORK - INFERENCE STAGE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 Configuration:")
    logger.info("-" * 80)
    logger.info(f"LLM Provider: {provider}")
    logger.info(f"LLM Model: {model}")
    logger.info("")
    logger.info(f"🔢 Embedder Provider: {ingestion_config.embedding_provider}")
    logger.info(f"🔢 Embedder Model: {ingestion_config.get_embedding_model()}")
    logger.info(f"🔢 Embedding Dimension: {ingestion_config.get_embedding_dimension()}")
    logger.info(f"🔢 Chunking Method: {ingestion_config.chunking_method}")
    logger.info(f"🔢 Chunk Size: {args.chunk_size or ingestion_config.chunk_size}")
    logger.info(f"🔢 Chunk Overlap: {args.chunk_overlap or ingestion_config.chunk_overlap}")
    logger.info("")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Documents: {args.docs}")
    logger.info(f"Top K Retrieval: {args.k}")
    logger.info(f"Debug Mode: {args.log_debug_information}")
    logger.info("")
    
    # Create experiment folder
    experiment_name = generate_experiment_name(model, args.experiment_name)
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment folder: {experiment_dir}")
    logger.info("")
    
    # Track experiment start time
    start_time = datetime.now()
    
    # Load golden dataset
    logger.info("Loading golden dataset...")
    try:
        golden_data = load_golden_dataset(args.dataset)
        logger.info(f"Loaded {len(golden_data)} questions from dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Create LLM
    logger.info("Initializing LLM...")
    llm = create_llm_provider(provider, model)
    
    # Create MemoryStore and ingest documents
    logger.info("Creating memory store and ingesting documents...")
    memory_store = MemoryStore()
    
    doc_ctx = DocumentCtx(
        tenant_id=args.tenant_id,
        owner_user_id=args.owner_user_id,
        document_id="master",
        visibility=args.visibility,
        embedding_version=ingestion_config.get_embedding_model()
    )
    
    total_chunks = load_and_ingest_documents(
        docs_path=args.docs,
        memory_store=memory_store,
        ctx=doc_ctx,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap
    )
    
    if total_chunks == 0:
        logger.error("No documents ingested. Exiting.")
        sys.exit(1)
    
    # Create retriever
    logger.info("Creating retriever...")
    retriever = MemoryRetrieverAdapter(
        store=memory_store,
        ctx=doc_ctx,
        top_k=args.k
    )
    
    # Build RAG workflow
    logger.info("Building RAG workflow...")
    rag_workflow = create_rag_workflow(llm, retriever, args.k)
    
    # Create prompt manager for debug mode
    prompt_manager = PromptManager() if args.log_debug_information else None
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize CSV file
    csv_path = experiment_dir / "results.csv"
    
    # Base fieldnames
    csv_fieldnames = [
        "question", "llm_answer", "gold_answer", "context", "bucket", "difficulty",
        "timestamp", "latency_seconds", "retrieved_context", "num_chunks_retrieved",
        "needs_retrieval", "rewritten_query", "embedding_model", "llm_model",
        "token_count_estimate", "cost_estimate"
    ]
    
    # Add debug columns if debug mode is enabled
    if args.log_debug_information:
        debug_fieldnames = [
            "debug_classification_latency",
            "debug_rewrite_latency", 
            "debug_retrieval_latency",
            "debug_generation_latency",
            "debug_retrieved_sources",
            "debug_retrieved_doc_ids",
            "debug_chunk_scores",
            "debug_original_query_length",
            "debug_rewritten_query_length",
            "debug_context_length",
            "debug_answer_length",
            "debug_workflow_state",
            "debug_classification_prompt",
            "debug_rewrite_prompt",
            "debug_generation_prompt"
        ]
        csv_fieldnames.extend(debug_fieldnames)
    
    # Track metrics
    total_latency = 0
    total_cost = 0
    error_count = 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        # Process each question
        for idx, item in enumerate(tqdm(golden_data, desc="Processing questions"), 1):
            try:
                question = item["question"]
                gold_answer = item.get("gold_answer", "")
                gt_context = item.get("context", "")  # Ground truth context from golden dataset
                bucket = item.get("bucket", "")
                difficulty = item.get("difficulty", "")
                
                # Run RAG workflow with debug tracking
                question_start = time.time()
                
                user_message = HumanMessage(content=question)
                initial_state = {
                    "messages": [user_message],
                    "question": question,
                    "rewritten_query": "",
                    "context": "",
                    "answer": "",
                    "needs_retrieval": False
                }
                
                # Extract debug info and run workflow
                workflow_output = extract_debug_information(
                    rag_workflow, 
                    initial_state, 
                    args.log_debug_information,
                    prompt_manager
                )
                result = workflow_output["result"]
                debug_data = workflow_output["debug"]
                
                latency = time.time() - question_start
                
                # Extract metrics
                answer = result.get("answer", "")
                context = result.get("context", "")
                rewritten_query = result.get("rewritten_query", "")
                needs_retrieval = result.get("needs_retrieval", False)
                
                # Count chunks (split by double newline)
                num_chunks = len([d for d in context.split("\n\n") if d.strip()])
                
                # Estimate tokens (rough: ~4 chars per token)
                input_tokens = (len(context) + len(question)) // 4
                output_tokens = len(answer) // 4
                total_tokens = input_tokens + output_tokens
                
                # Estimate cost
                cost = estimate_cost(model, input_tokens, output_tokens)
                
                # Build base row
                row = {
                    "question": question,
                    "llm_answer": answer,
                    "gold_answer": gold_answer,
                    "context": format_ground_truth_context(gt_context),  # Ground truth context from golden dataset (formatted)
                    "bucket": bucket,
                    "difficulty": difficulty,
                    "timestamp": datetime.now().isoformat(),
                    "latency_seconds": f"{latency:.3f}",
                    "retrieved_context": context,  # Actually retrieved context (full, not truncated)
                    "num_chunks_retrieved": num_chunks,
                    "needs_retrieval": needs_retrieval,
                    "rewritten_query": rewritten_query,
                    "embedding_model": ingestion_config.get_embedding_model(),
                    "llm_model": model,
                    "token_count_estimate": total_tokens,
                    "cost_estimate": f"{cost:.6f}"
                }
                
                # Add debug columns if enabled
                if args.log_debug_information:
                    row.update({
                        "debug_classification_latency": f"{debug_data.get('classification_latency', 0):.4f}",
                        "debug_rewrite_latency": f"{debug_data.get('rewrite_latency', 0):.4f}",
                        "debug_retrieval_latency": f"{debug_data.get('retrieval_latency', 0):.4f}",
                        "debug_generation_latency": f"{debug_data.get('generation_latency', 0):.4f}",
                        "debug_retrieved_sources": debug_data.get('retrieved_sources', ''),
                        "debug_retrieved_doc_ids": debug_data.get('retrieved_doc_ids', ''),
                        "debug_chunk_scores": debug_data.get('chunk_scores', ''),
                        "debug_original_query_length": debug_data.get('original_query_length', 0),
                        "debug_rewritten_query_length": debug_data.get('rewritten_query_length', 0),
                        "debug_context_length": debug_data.get('context_length', 0),
                        "debug_answer_length": debug_data.get('answer_length', 0),
                        "debug_workflow_state": debug_data.get('workflow_state', ''),
                        "debug_classification_prompt": debug_data.get('classification_prompt', ''),
                        "debug_rewrite_prompt": debug_data.get('rewrite_prompt', ''),
                        "debug_generation_prompt": debug_data.get('generation_prompt', '')
                    })
                
                writer.writerow(row)
                csvfile.flush()  # Ensure data is written incrementally
                
                # Update totals
                total_latency += latency
                total_cost += cost
                
            except Exception as e:
                logger.error(f"Error processing question {idx}: {e}")
                error_count += 1
                # Write error row
                row = {
                    "question": item.get("question", ""),
                    "llm_answer": f"ERROR: {str(e)}",
                    "gold_answer": item.get("gold_answer", ""),
                    "context": format_ground_truth_context(item.get("context", "")),  # GT context from golden dataset (formatted)
                    "bucket": item.get("bucket", ""),
                    "difficulty": item.get("difficulty", ""),
                    "timestamp": datetime.now().isoformat(),
                    "latency_seconds": "0",
                    "retrieved_context": "",  # No retrieval on error
                    "num_chunks_retrieved": 0,
                    "needs_retrieval": False,
                    "rewritten_query": "",
                    "embedding_model": ingestion_config.get_embedding_model(),
                    "llm_model": model,
                    "token_count_estimate": 0,
                    "cost_estimate": "0"
                }
                
                # Add empty debug columns if enabled
                if args.log_debug_information:
                    row.update({
                        "debug_classification_latency": "0",
                        "debug_rewrite_latency": "0",
                        "debug_retrieval_latency": "0",
                        "debug_generation_latency": "0",
                        "debug_retrieved_sources": "",
                        "debug_retrieved_doc_ids": "",
                        "debug_chunk_scores": "",
                        "debug_original_query_length": 0,
                        "debug_rewritten_query_length": 0,
                        "debug_context_length": 0,
                        "debug_answer_length": 0,
                        "debug_workflow_state": "ERROR",
                        "debug_classification_prompt": "",
                        "debug_rewrite_prompt": "",
                        "debug_generation_prompt": ""
                    })
                
                writer.writerow(row)
                csvfile.flush()
    
    # Track experiment end time
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Calculate statistics
    num_successful = len(golden_data) - error_count
    avg_latency = total_latency / num_successful if num_successful > 0 else 0
    
    # Write metadata file
    metadata_path = experiment_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("Experiment Configuration\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment Name: {experiment_name}\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n")
        f.write("\n")
        
        f.write("Dataset\n")
        f.write("-" * 60 + "\n")
        f.write(f"Dataset Path: {args.dataset}\n")
        f.write(f"Number of Questions: {len(golden_data)}\n")
        f.write(f"Documents Path: {args.docs}\n")
        f.write(f"Total Chunks Ingested: {total_chunks}\n")
        f.write("\n")
        
        f.write("Embedding Configuration\n")
        f.write("-" * 60 + "\n")
        f.write(f"Provider: {ingestion_config.embedding_provider}\n")
        f.write(f"Model: {ingestion_config.get_embedding_model()}\n")
        f.write(f"Chunking Method: {ingestion_config.chunking_method}\n")
        f.write(f"Chunk Size: {args.chunk_size or ingestion_config.chunk_size}\n")
        f.write(f"Chunk Overlap: {args.chunk_overlap or ingestion_config.chunk_overlap}\n")
        f.write("\n")
        
        f.write("LLM Configuration\n")
        f.write("-" * 60 + "\n")
        f.write(f"Provider: {provider}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Temperature: 0\n")
        f.write("\n")
        
        f.write("Retrieval Configuration\n")
        f.write("-" * 60 + "\n")
        f.write(f"Top K: {args.k}\n")
        f.write(f"Tenant ID: {args.tenant_id}\n")
        f.write(f"Visibility: {args.visibility}\n")
        f.write(f"Debug Mode: {args.log_debug_information}\n")
        f.write("\n")
        
        f.write("Git Versions\n")
        f.write("-" * 60 + "\n")
        f.write(f"Main Repo Commit: {get_git_commit('.')}\n")
        f.write(f"Ingestion Module Commit: {get_git_commit('src/ingestion')}\n")
        f.write(f"RAG Module Commit: {get_git_commit('src/rag-module')}\n")
        f.write("\n")
        
        f.write("Summary Statistics\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Questions: {len(golden_data)}\n")
        f.write(f"Successful: {num_successful}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"Average Latency: {avg_latency:.3f}s\n")
        f.write(f"Total Latency: {total_latency:.2f}s\n")
        f.write(f"Total Cost Estimate: ${total_cost:.4f}\n")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {csv_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Total questions: {len(golden_data)}")
    logger.info(f"Successful: {num_successful}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Average latency: {avg_latency:.3f}s")
    logger.info(f"Total cost estimate: ${total_cost:.4f}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework - Run inference on golden datasets"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSON file or folder containing golden dataset files"
    )
    parser.add_argument(
        "--docs",
        default="docs",
        help="Documents directory (default: docs)"
    )
    
    # Experiment configuration
    parser.add_argument(
        "--experiment-name",
        help="Optional custom experiment name (default: auto-generated)"
    )
    
    # RAG configuration
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of documents to retrieve (default: 8)"
    )
    
    # Ingestion configuration
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (uses config default if not set)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap (uses config default if not set)"
    )
    
    # Document context configuration
    parser.add_argument(
        "--tenant-id",
        default="default-tenant",
        help="Tenant ID for multi-tenancy (default: default-tenant)"
    )
    parser.add_argument(
        "--owner-user-id",
        default="admin",
        help="Document owner user ID (default: admin)"
    )
    parser.add_argument(
        "--visibility",
        choices=["org", "private"],
        default="org",
        help="Document visibility scope (default: org)"
    )
    
    # Debug configuration
    parser.add_argument(
        "--log-debug-information",
        action="store_true",
        help="Enable debug mode to log intermediate steps and add debug columns to CSV output"
    )
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args)
    except KeyboardInterrupt:
        logger.info("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

