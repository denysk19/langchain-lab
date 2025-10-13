#!/usr/bin/env python3
"""
RAG Evaluation Framework - Inference Stage

Runs the RAG system through golden datasets and stores comprehensive results
with metadata tracking for experiment comparison and analysis.
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
    level=logging.WARNING,  # Suppress verbose logs during evaluation
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAG_EVAL")


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
    model = os.getenv("MODEL", "gpt-4o-mini")
    
    logger.info("=" * 80)
    logger.info("RAG EVALUATION FRAMEWORK - INFERENCE STAGE")
    logger.info("=" * 80)
    logger.info(f"LLM Provider: {provider}, Model: {model}")
    logger.info(f"Embedding: {ingestion_config.embedding_provider} - {ingestion_config.get_embedding_model()}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Documents: {args.docs}")
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
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize CSV file
    csv_path = experiment_dir / "results.csv"
    csv_fieldnames = [
        "question", "llm_answer", "gold_answer", "context", "bucket", "difficulty",
        "timestamp", "latency_seconds", "retrieved_context", "num_chunks_retrieved",
        "needs_retrieval", "rewritten_query", "embedding_model", "llm_model",
        "token_count_estimate", "cost_estimate"
    ]
    
    # Track metrics
    total_latency = 0
    total_cost = 0
    error_count = 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        
        # Process each question
        for idx, item in enumerate(tqdm(golden_data, desc="Processing questions"), 1):
            try:
                question = item["question"]
                gold_answer = item.get("gold_answer", "")
                gt_context = item.get("context", "")  # Ground truth context from golden dataset
                bucket = item.get("bucket", "")
                difficulty = item.get("difficulty", "")
                
                # Run RAG workflow
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
                
                result = rag_workflow.invoke(initial_state)
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
                
                # Truncate retrieved context if too long for CSV (keep first 1000 chars)
                retrieved_context_truncated = context[:1000] + "..." if len(context) > 1000 else context
                
                # Write row
                row = {
                    "question": question,
                    "llm_answer": answer,
                    "gold_answer": gold_answer,
                    "context": gt_context,  # Ground truth context from golden dataset
                    "bucket": bucket,
                    "difficulty": difficulty,
                    "timestamp": datetime.now().isoformat(),
                    "latency_seconds": f"{latency:.3f}",
                    "retrieved_context": retrieved_context_truncated,  # Actually retrieved context
                    "num_chunks_retrieved": num_chunks,
                    "needs_retrieval": needs_retrieval,
                    "rewritten_query": rewritten_query,
                    "embedding_model": ingestion_config.get_embedding_model(),
                    "llm_model": model,
                    "token_count_estimate": total_tokens,
                    "cost_estimate": f"{cost:.6f}"
                }
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
                    "context": item.get("context", ""),  # GT context from golden dataset
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

