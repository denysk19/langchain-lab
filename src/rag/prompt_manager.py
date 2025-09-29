"""Prompt management system for RAG workflow."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompts from configuration files with variant support."""
    
    def __init__(self, config_path: Optional[str] = None, variant: str = "default"):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "prompts.yaml"
        self.variant = variant
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            prompts = config.get('prompts', {})
            variants = config.get('variants', {})
            
            # Apply variant overrides
            if self.variant != "default" and self.variant in variants:
                for section, variant_prompts in variants[self.variant].items():
                    if section in prompts:
                        prompts[section].update(variant_prompts)
                    else:
                        prompts[section] = variant_prompts
                        
            logger.info(f"Loaded prompts from {self.config_path} (variant: {self.variant})")
            return prompts
            
        except Exception as e:
            logger.error(f"Failed to load prompts from {self.config_path}: {e}")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """Fallback default prompts if config file fails to load."""
        return {
            "classification": {
                "system": "Classify if the question needs document retrieval. Reply 'RETRIEVE' or 'DIRECT' only.",
                "human_template": "Question: {question}"
            },
            "query_rewrite": {
                "system": "Rewrite the question as a standalone query. Return only the rewritten query.",
                "human_template": "Question: {question}"
            },
            "answer_generation": {
                "retrieval_based": {
                    "system": "Answer using the provided information.",
                    "human_template": "Information: {context}\n\nQuestion: {question}"
                },
                "direct_answer": {
                    "system": "Answer the question helpfully.",
                    "human_template": "Question: {question}"
                }
            }
        }
    
    def get_classification_prompts(self, context: str, question: str) -> Tuple[str, str]:
        """Get classification prompts with filled templates."""
        config = self.prompts.get('classification', {})
        
        system_prompt = config.get('system', '')
        human_template = config.get('human_template', 'Question: {question}')
        human_prompt = human_template.format(context=context, question=question)
        
        return system_prompt, human_prompt
    
    def get_rewrite_prompts(self, context: str, question: str) -> Tuple[str, str]:
        """Get query rewrite prompts with filled templates."""
        config = self.prompts.get('query_rewrite', {})
        
        system_prompt = config.get('system', '')
        human_template = config.get('human_template', 'Question: {question}')
        human_prompt = human_template.format(context=context, question=question)
        
        return system_prompt, human_prompt
    
    def get_generation_prompts(self, mode: str, context: str = "", 
                             conversation_context: str = "", question: str = "") -> Tuple[str, str]:
        """Get answer generation prompts."""
        if mode == "retrieval_based":
            config = self.prompts.get('answer_generation', {}).get('retrieval_based', {})
        else:
            config = self.prompts.get('answer_generation', {}).get('direct_answer', {})
        
        system_prompt = config.get('system', '')
        human_template = config.get('human_template', 'Question: {question}')
        
        # Format conversation context
        conv_context = f"Recent conversation: {conversation_context}" if conversation_context else ""
        
        human_prompt = human_template.format(
            context=context,
            conversation_context=conv_context,
            question=question
        )
        
        return system_prompt, human_prompt
    
    def get_summarization_prompt(self, conversation: str, current_summary: str = None) -> str:
        """Get summarization prompt."""
        if current_summary:
            template = self.prompts.get('summarization', {}).get('update_existing', '')
            return template.format(current_summary=current_summary, conversation=conversation)
        else:
            template = self.prompts.get('summarization', {}).get('create_new', '')
            return template.format(conversation=conversation)
    
    def reload_prompts(self) -> None:
        """Reload prompts from configuration file."""
        self.prompts = self._load_prompts()
        logger.info("Prompts reloaded from configuration")
    
    def get_available_variants(self) -> list[str]:
        """Get list of available prompt variants."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            variants = list(config.get('variants', {}).keys())
            return ['default'] + variants
        except Exception:
            return ['default']
