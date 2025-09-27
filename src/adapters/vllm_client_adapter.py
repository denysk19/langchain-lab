"""
Optional vLLM client adapter for direct API calls.
This is an alternative to using ChatOpenAI with base_url.
"""

import json
from typing import Optional

import requests
from langchain_core.runnables import RunnableLambda


class SimpleVLLMClient:
    """Simple vLLM client that calls OpenAI-compatible API directly."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            })
        else:
            self.session.headers.update({
                "Content-Type": "application/json"
            })
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1024, 
        temperature: float = 0.0, 
        **kwargs
    ) -> str:
        """Generate text using vLLM OpenAI-compatible endpoint."""
        
        # Convert single prompt to OpenAI chat format
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": kwargs.get("model", "default"),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["model"]:  # model already handled
                payload[key] = value
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected vLLM API response format: {e}")


def make_vllm_runnable(client: SimpleVLLMClient, **default_kwargs) -> RunnableLambda:
    """
    Create a LangChain-compatible runnable from a vLLM client.
    
    This is an alternative approach to using ChatOpenAI with base_url.
    """
    
    def _call_vllm(prompt: str) -> str:
        """Call vLLM client and return the response."""
        return client.generate(prompt, **default_kwargs)
    
    return RunnableLambda(_call_vllm)


# Example usage (not used by default in chat.py):
"""
# In chat.py, you could use this instead of ChatOpenAI for vLLM:

def create_vllm_with_adapter():
    client = SimpleVLLMClient(
        base_url=os.getenv("VLLM_BASE_URL"),
        api_key=os.getenv("VLLM_API_KEY")
    )
    
    return make_vllm_runnable(
        client,
        model=os.getenv("VLLM_MODEL", "default"),
        temperature=0.0,
        max_tokens=1024
    )

# Then in the provider factory:
elif provider == "vllm":
    return create_vllm_with_adapter()
"""

