"""
Factory function for initializing LLM providers based on environment configuration.
"""

import os
from dotenv import load_dotenv
from llm.base import BaseLLMProvider
from llm.mock_provider import MockLLMProvider
from llm.groq_provider import GroqLLMProvider

load_dotenv()


def get_llm_provider() -> BaseLLMProvider:
    """
    Instantiate the appropriate LLM provider.
    Falls back gracefully to MockLLMProvider if Groq is not configured or fails.
    """
    provider_type = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    
    if provider_type == "groq":
        try:
            return GroqLLMProvider()
        except ValueError as e:
            # Fallback to mock provider with notification
            print(f"Warning: {e} Falling back to MockLLMProvider.")
            return MockLLMProvider()
            
    return MockLLMProvider()
